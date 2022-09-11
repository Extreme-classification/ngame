import torch
import math
import os
from .transformer_layer import STransformer
import torch.nn.functional as F


class Network(torch.nn.Module):

    def __init__(self, transformer, config=None, projection_dims=-1, metric='cosine', device="cuda"):
        super(Network, self).__init__()
        self.encoder = self._construct_transform(transformer, config)
        if projection_dims != -1:
            self.transform = torch.nn.Linear(self.encoder.repr_dims, projection_dims)
            self.representation_dims = projection_dims
        else:
            self.transform = torch.nn.Identity()
            self.representation_dims = self.encoder.repr_dims
        self.metric = metric
        self.device = torch.device(device)
        print(self.encoder)

    def _construct_transform(self, transformer, config):
        return STransformer(transformer)

    @property
    def representation_dims(self):
        return self._repr_dims

    @representation_dims.setter
    def representation_dims(self, dims):
        self._repr_dims = dims

    def encode_document(self, data, *args):
        return self.encode(data['ind'], data['mask'])

    def encode_label(self, data, *args):
        return self.encode(data['ind'], data['mask'])

    def encode(self, x_ind, x_mask):
        return self.encoder(x_ind.to(self.device), x_mask.to(self.device))

    def similarity(self, ip_rep, op_rep):
        #  Units vectors in case of cosine similarity
        if self.metric == 'cosine':
            ip_rep = F.normalize(ip_rep, dim=1)
            op_rep = F.normalize(op_rep, dim=1)
        return ip_rep @ op_rep.T

    def forward(self, data, *args):
        ip_rep = self.encode(data['ip_ind'], data['ip_mask'])
        op_rep = self.encode(data['op_ind'], data['op_mask'])
        return self.similarity(ip_rep, op_rep)

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
        return f"(Transform): {self.transform}"