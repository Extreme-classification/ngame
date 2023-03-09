import torch
import math
import models.transform_layer as transform_layer
from .transformer_layer import STransformer
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import os
import models.linear_layer as linear_layer


def _to_device(x, device):
    if x is None:
        return None
    elif isinstance(x, (tuple, list)):
        out = []
        for item in x:
            out.append(_to_device(item, device))
        return out
    else:
        return x.to(device)



class DeepXMLBase(nn.Module):
    """DeepXMLBase: Base class for DeepXML architecture
    * Identity op as classifier by default
    (derived class should implement it's own classifier)
    * embedding and classifier shall automatically transfer
    the vector to the appropriate device
    Arguments:
    ----------
    vocabulary_dims: int
        number of tokens in the vocabulary
    embedding_dims: int
        size of word/token representations
    trans_config: list of strings
        configuration of the transformation layer
    padding_idx: int, default=0
        padding index in words embedding layer
    """

    def __init__(self, config, device="cuda"):
        super(DeepXMLBase, self).__init__()
        self.encoder = self._construct_transform(config)
        self.classifier = self._construct_classifier()
        self.device = torch.device(device)

    def _construct_classifier(self):
        return nn.Identity()

    def _construct_transform(self, trans_config):
        if trans_config is None:
            return None
        return transform_layer.Transform(
            transform_layer.get_functions(trans_config))

    @property
    def representation_dims(self):
        return self._repr_dims

    @representation_dims.setter
    def representation_dims(self, dims):
        self._repr_dims = dims

    def encode(self, x):
        """Forward pass
        * Assumes features are dense if x_ind is None
        Arguments:
        -----------
        x: tuple
            torch.FloatTensor or None
                (sparse features) contains weights of features as per x_ind or
                (dense features) contains the dense representation of a point
            torch.LongTensor or None
                contains indices of features (sparse or seqential features)
        Returns
        -------
        out: logits for each label
        """
        return self.transform(
            _to_device(x, self.device))

    def forward(self, batch_data, *args):
        """Forward pass
        * Assumes features are dense if X_w is None
        * By default classifier is identity op
        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep.
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None
        Returns
        -------
        out: logits for each label
        """
        return self.classifier(
            self.encode(batch_data['X'], batch_data['X_ind']))

    def initialize(self, x):
        """Initialize embeddings from existing ones
        Parameters:
        -----------
        word_embeddings: numpy array
            existing embeddings
        """
        self.transform.initialize(x)

    # def to(self):
    #     """Send layers to respective devices
    #     """
    #     self.transform.to()
    #     self.classifier.to()

    def purge(self, fname):
        if os.path.isfile(fname):
            os.remove(fname)

    @property
    def num_params(self, ignore_fixed=False):
        if ignore_fixed:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    @property
    def model_size(self):  # Assumptions: 32bit floats
        return self.num_params * 4 / math.pow(2, 20)

    def __repr__(self):
        return f"{self.embeddings}\n(Transform): {self.transform}"



class SiameseXML(DeepXMLBase):
    """
    Siamese to embed document and labels together
    * Allows different or same embeddings for documents and labels
    * Allows different or same transformation for documents and labels
    """
    def __init__(self, params, device="cuda"):
        super(SiameseXML, self).__init__(None)
        self.share_weights = params.share_weights
        self.metric = params.metric
        config_dict = transform_layer.fetch_json(
            params.arch, params)
        self.representation_dims = int(config_dict['representation_dims'])

        #  Network to embed document
        self.encoder = self._construct_transform(config_dict['encoder'])

        if not self.share_weights:
            self.encoder_lbl = self._construct_transform(
                config_dict['encoder_lbl'])
        else:
            self._create_shared_net()

        self.transform_doc = self._construct_transform(
            config_dict['transform'])
        self.transform_lbl = self._construct_transform(
            config_dict['transform_lbl'])

    def _create_shared_net(self):
        self.encoder_lbl = self.encoder
        
    def encode(self, x, x_ind=None, return_coarse=False):
        return self.encode_document(x, x_ind, return_coarse)

    def encode_document(self, x=None, ind=None,
                        mask=None, return_coarse=False):
        """Forward pass
        * Assumes features are dense if x_ind is None
        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        return_coarse: boolean, optional (default=False)
            Return coarse features or not
        Returns
        -------
        out: logits for each label
        """
        encoding = self.encoder.encode(
            _to_device((ind, mask), self.device))
        if not return_coarse:
            encoding = self.transform_doc(encoding)
        return encoding

    def encode_label(self, x=None, ind=None, mask=None, return_coarse=False):
        """Forward pass
        * Assumes features are dense if x_ind is None
        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        return_coarse: boolean, optional (default=False)
            Return coarse features or not
        Returns
        -------
        out: logits for each label
        """
        encoding = self.encoder_lbl.encode(
            _to_device((ind, mask), self.device))
        if not return_coarse:
            encoding = self.transform_lbl(encoding)
        return encoding

    def similarity(self, doc_rep, lbl_rep):
        #  Units vectors in case of cosine similarity
        if self.metric == 'cosine':
            doc_rep = F.normalize(doc_rep, dim=1)
            lbl_rep = F.normalize(lbl_rep, dim=1)
        return doc_rep @ lbl_rep.T

    def forward(self, data, *args, **kwargs):
        ip_rep = self.encode_document(
            ind=data['ip_ind'],
            mask=data['ip_mask'])
        op_rep = self.encode_label(
            ind=data['op_ind'],
            mask=data['op_mask'])
        return self.similarity(ip_rep, op_rep), ip_rep

    # def to(self):
    #     """Send layers to respective devices
    #     """
    #     self.transform_fine_document.to()
    #     self.transform_fine_label.to()
    #     self.document_net.to()
    #     self.label_net.to()

    def initialize(self, x):
        """Initialize parameters from existing ones
        
        Parameters:
        -----------
        x: numpy array
            existing parameters
        """
        self.encoder.initialize(x)
        if not self.share_weights:
            self.encoder_lbl.initialize(x)

    def save_intermediate_model(self, fname):
        out = {}
        if not self.share_weights:
            out['encoder'] = self.encoder.state_dict()
            out['encoder_lbl'] = self.encoder_lbl.state_dict()
        else:
            out = {'encoder': self.encoder.state_dict()}
        torch.save(out, fname)

    def load_intermediate_model(self, fname):
        out = pickle.load(open(fname, 'rb'))
        if not self.share_weights:
            self.encoder.load_state_dict(out['encoder'])
            self.encoder_lbl.load_state_dict(out['encoder_lbl'])
        else:
            self.encoder.load_state_dict(out['encoder'])

    def named_parameters(self, recurse=True, return_shared=False):
        if self.share_weights and not return_shared:
            # Assuming label_net is a copy of document_net
            for name, param in super().named_parameters(recurse=recurse):
                if 'encoder_lbl' not in name:
                    yield name, param
        else:
            for name, param in super().named_parameters(recurse=recurse):
                yield name, param

    def parameters(self, recurse=True, return_shared=False):
        if self.share_weights and not return_shared:
            # Assuming encoder_lbl is a copy of encoder
            for name, param in super().named_parameters(recurse=recurse):
                if 'encoder_lbl' not in name:
                    yield param
        else:
            for name, param in self.named_parameters(recurse=recurse):
                yield param

    @property
    def modules_(self, return_shared=False):
        out = OrderedDict()
        for k, v in self._modules.items():
            if not return_shared and self.share_weights and 'encoder_lbl' in k:
                continue
            out[k] = v
        return out

    def __repr__(self):
        s = f"{self.__class__.__name__} (Weights shared: {self.share_weights})"
        s += f"\n(Encoder): {self.encoder}\n"
        s += f"(Transform Doc): {self.transform_doc} \n"
        s += f"(EncoderLabel): {self.encoder_lbl}\n"
        s += f"(Transform Label): {self.transform_lbl} \n"
        return s


class DeepXMLSS(DeepXMLBase):
    """DeepXMLs: DeepXML architecture to be trained with
                 a shared shortlist
    * Allows additional transform layer for features
    """

    def __init__(self, params):
        self.num_labels = params.num_labels
        self.label_padding_index = params.label_padding_index
        config_dict = transform_layer.fetch_json(
            params.arch, params)
        self.representation_dims = int(
            config_dict['representation_dims'])
        self.metric = params.metric
        super(DeepXMLSS, self).__init__(config_dict['encoder'])
        if params.freeze_intermediate:
            for params in self.encoder.parameters():
                params.requires_grad = False
        self.transform = self._construct_transform(
            config_dict['transform_doc'])

    def save_intermediate_model(self, fname):
        out = {'encoder': self.encoder.state_dict()
            }
        torch.save(out, fname)

    def load_intermediate_model(self, fname):
        self.encoder.load_state_dict(torch.load(fname)['encoder'])

    def encode_transform(self, x):
        """Forward pass (assumes input is intermediate computation)
        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        Returns
        -------
        out: torch.FloatTensor
            encoded x with fine encoder
        """
        return self.transform(_to_device(x, self.device))

    def encode_document(self, x, ind=None, mask=None, return_coarse=False):
        return self.encode(x, ind, mask, return_coarse)

    def encode_label(self, x, ind=None, mask=None, return_coarse=False):
        return self.encode(x, ind, mask, return_coarse)

    def encode(self, x=None, ind=None, mask=None, bypass_fine=False):
        #TODO: Implement stuff for non-shared arch
        """Forward pass
        * Assumes features are dense if x_ind is None
        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        bypass_fine: boolean, optional (default=False)
            Return coarse features or not
        Returns
        -------
        out: logits for each label
        """
        encoding = self.encoder.encode(
            _to_device((ind, mask), self.device))
        return encoding if bypass_fine else self.transform(encoding)

    def forward(self, batch_data, bypass_coarse=False):
        """Forward pass
        * Assumes features are dense if X_w is None
        * By default classifier is identity op
        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep.
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None
        Returns
        -------
        out: logits for each label
        """
        if bypass_coarse:
            return self.classifier(
                self.encode_transform(batch_data['X']), batch_data['Y_s'])
        else:
            return self.classifier(
                self.encode(ind=batch_data['ind'], mask=batch_data['mask']),
                batch_data['Y_s'])

    def _construct_classifier(self):
        offset = 0

        if self.label_padding_index:
            offset += 1
        # last one is padding index
        if self.metric == 'cosine':
            return linear_layer.UNSSparseLinear(
                input_size=self.representation_dims,
                output_size=self.num_labels + offset,
                padding_idx=self.label_padding_index)
        else:
            return linear_layer.SparseLinear(
                input_size=self.representation_dims,
                output_size=self.num_labels + offset,
                padding_idx=self.label_padding_index,
                bias=True)

    # def to(self):
    #     """Send layers to respective devices
    #     """
    #     self.transform_fine.to()
    #     super().to()

    def initialize_classifier(self, weight, bias=None):
        """Initialize classifier from existing weights
        Arguments:
        -----------
        weight: numpy.ndarray
        bias: numpy.ndarray or None, optional (default=None)
        """
        self.classifier.weight.data.copy_(torch.from_numpy(weight))
        if bias is not None:
            self.classifier.bias.data.copy_(
                torch.from_numpy(bias).view(-1, 1))

    def get_clf_weights(self):
        """Get classifier weights
        """
        return self.classifier.get_weights()

    def __repr__(self):
        s = f"{self.encoder}\n"
        s += f"(Transform): {self.transform}"
        s += f"\n(Classifier): {self.classifier}\n"
        return s

# class Network(torch.nn.Module):

#     def __init__(self, transformer, config=None, projection_dims=-1, metric='cosine', device="cuda"):
#         super(Network, self).__init__()
#         self.encoder = self._construct_transform(transformer, config)
#         if projection_dims != -1:
#             self.transform = torch.nn.Linear(self.encoder.repr_dims, projection_dims)
#             self.representation_dims = projection_dims
#         else:
#             self.transform = torch.nn.Identity()
#             self.representation_dims = self.encoder.repr_dims
#         self.metric = metric
#         self.device = torch.device(device)
#         print(self.encoder)

#     def _construct_transform(self, transformer, config):
#         return STransformer(transformer)

#     @property
#     def representation_dims(self):
#         return self._repr_dims

#     @representation_dims.setter
#     def representation_dims(self, dims):
#         self._repr_dims = dims

#     def encode_document(self, data, *args):
#         return self.encode(data['ind'], data['mask'])

#     def encode_label(self, data, *args):
#         return self.encode(data['ind'], data['mask'])

#     def encode(self, x_ind, x_mask):
#         return self.encoder(x_ind.to(self.device), x_mask.to(self.device))

#     def similarity(self, ip_rep, op_rep):
#         #  Units vectors in case of cosine similarity
#         if self.metric == 'cosine':
#             ip_rep = F.normalize(ip_rep, dim=1)
#             op_rep = F.normalize(op_rep, dim=1)
#         return ip_rep @ op_rep.T

#     def forward(self, data, *args):
#         ip_rep = self.encode(data['ip_ind'], data['ip_mask'])
#         op_rep = self.encode(data['op_ind'], data['op_mask'])
#         return self.similarity(ip_rep, op_rep), ip_rep

#     @property
#     def num_params(self, ignore_fixed=False):
#         if ignore_fixed:
#             return sum(p.numel() for p in self.parameters() if p.requires_grad)
#         else:
#             return sum(p.numel() for p in self.parameters())

#     @property
#     def model_size(self):  # Assumptions: 32bit floats
#         return self.num_params * 4 / math.pow(2, 20)

#     def __repr__(self):
#         return f"(Transform): {self.transform}"