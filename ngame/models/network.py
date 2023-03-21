import torch
import math
import models.transform_layer as transform_layer
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import os
import models.linear_layer as linear_layer


def construct_network(args):
    if args.network_type == 'siamese':
        if args.sampling_type == "implicit":
            net = SiameseXMLIS(args)
        elif args.sampling_type == 'explicit':
            raise NotImplementedError("")
        else:
            raise NotImplementedError("")
    elif args.network_type == 'xc':
        if args.sampling_type == "implicit":
            net = DeepXMLIS(args)
        elif args.sampling_type == 'explicit':
            raise NotImplementedError("")
        else:
            raise NotImplementedError("")
    else:
        raise NotImplementedError("")
    return net


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


class SiameseXMLIS(DeepXMLBase):
    """
    Siamese to embed document and labels together
    * Allows different or same embeddings for documents and labels
    * Allows different or same transformation for documents and labels
    """
    def __init__(self, params, device="cuda"):
        super(SiameseXMLIS, self).__init__(None)
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
        
    def encode_document(self, x, return_encoder_rep=False):
        """Forward pass
        * Assumes features are dense if x_ind is None
        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        return_encoder_rep: boolean, optional (default=False)
            Return encoder representation or not
        Returns
        -------
        out: logits for each label
        """
        encoding = self.encoder.encode(_to_device(x, self.device))
        if not return_encoder_rep:
            encoding = self.transform_doc(encoding)
        return encoding

    def encode_label(self, x, return_encoder_rep=False):
        """Forward pass
        * Assumes features are dense if x_ind is None
        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        return_encoder_rep: boolean, optional (default=False)
            Return coarse features or not
        Returns
        -------
        out: logits for each label
        """
        encoding = self.encoder_lbl.encode(_to_device(x, self.device))
        if not return_encoder_rep:
            encoding = self.transform_lbl(encoding)
        return encoding

    def similarity(self, doc_rep, lbl_rep):
        #  Units vectors in case of cosine similarity
        if self.metric == 'cosine':
            doc_rep = F.normalize(doc_rep, dim=1)
            lbl_rep = F.normalize(lbl_rep, dim=1)
        return doc_rep @ lbl_rep.T

    def forward(self, batch_data, *args, **kwargs):
        ip_rep = self.encode_document(batch_data['X'])
        op_rep = self.encode_label(batch_data['Z'])
        return self.similarity(ip_rep, op_rep), ip_rep

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


class DeepXMLIS(DeepXMLBase):
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
        super(DeepXMLIS, self).__init__(config_dict['encoder'])
        if params.freeze_encoder:
            for params in self.encoder.parameters():
                params.requires_grad = False
        self.transform = self._construct_transform(
            config_dict['transform_doc'])

    def save_intermediate_model(self, fname):
        out = {'encoder': self.encoder.state_dict()}
        torch.save(out, fname)

    def load_intermediate_model(self, fname):
        self.encoder.load_state_dict(torch.load(fname)['encoder'])

    def _encode(self, x, *args, **kwargs):
        return self.encoder.encode(_to_device(x, self.device))

    def _encode_transform(self, x, *args, **kwargs):
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

    def encode_document(self, x, ret_encoder_rep=False):
        return self.encode(x, ret_encoder_rep)

    def encode_label(self, x, ret_encoder_rep=False):
        return self.encode(x, ret_encoder_rep)

    def encode(self, x, ret_encoder_rep=False):
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
        ret_encoder_rep: boolean, optional (default=False)
            Return coarse features or not
        Returns
        -------
        out: logits for each label
        """
        encoding = self.encoder.encode(
            _to_device(x, self.device))
        return encoding if ret_encoder_rep else self.transform(encoding)

    def forward(self, batch_data, bypass_encoder=False):
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
        if bypass_encoder:
            X = self._encode_transform(batch_data['X'])
        else:
            X = self.encode(batch_data['X'])
        return self.classifier(X, batch_data['Y_s']), X

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
        s = f"(Encoder): {self.encoder}\n"
        s += f"(Transform): {self.transform}"
        s += f"\n(Classifier): {self.classifier}\n"
        return s

