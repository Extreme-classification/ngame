import torch
import pickle
import os
import numpy as np
from .features import construct as construct_f
from .labels import construct as construct_l


class DatasetTensor(torch.utils.data.Dataset):
    """Dataset to load and use sparse/dense matrix
    Support npz, pickle, npy or libsvm file format
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname: str
        file name file (libsvm or npy or npz or pkl)
        will use 'X' key in case of pickle
    data: scipy.sparse or np.ndarray, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
    normalize: bool, optional, default=True
        Normalize the rows to unit norm
    """

    def __init__(self, data_dir, fname, data=None, indices=None,
                 normalize=True, _type='sparse'):
        self.data = self.construct(
            data_dir, fname, data, indices, normalize, _type)

    def construct(self, data_dir, fname, data, indices, normalize, _type):
        data = construct_f(data_dir, fname, data, normalize, _type)
        if indices is not None:
            indices = np.loadtxt(indices, dtype=np.int64)
            data._index_select(indices)
        return data

    def __len__(self):
        return self.num_instances

    @property
    def num_instances(self):
        return self.data.num_instances

    def __getitem__(self, index):
        """Get data for a given index
        Arguments
        ---------
        index: int
            data for this index
        Returns
        -------
        features: tuple
            feature indices and their weights
        """
        return self.data[index]
