from xclib.data.features import DenseFeatures, SparseFeatures, FeaturesBase
from xclib.data import data_utils
import os
import numpy as np


def construct(data_dir, fname, X=None, normalize=False, _type='sparse'):
    """Construct feature class based on given parameters
    Arguments
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    X: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the data or not
    _type: str, optional, default=sparse
        -sparse
        -dense
        -sequential
    """
    if _type == 'sparse':
        return _SparseFeatures(data_dir, fname, X, normalize)
    elif _type == 'dense':
        return DenseFeatures(data_dir, fname, X, normalize)
    elif _type == 'sequential':
        return SeqFeatures(data_dir, fname, X)
    else:
        raise NotImplementedError("Unknown feature type")


class SeqFeatures(SparseFeatures):
    """Class for sparse features
    * Difference: treat 0 as padding index
    
    Arguments
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    X: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the data or not
    """

    def __init__(self, data_dir, fname, X=None):
        super().__init__(data_dir, fname, X)

    def load(self, data_dir, fname, X):
        """
        Load data (to be implemented for specific features)
        """
        if X is not None:
            return X
        else:
            f_ids, f_mask = fname.split(",")
            X = np.load(
                os.path.join(data_dir, f_ids),
                mmap_mode='r')
            X_mask = np.load(
                os.path.join(data_dir, f_mask),
                mmap_mode='r')
            return X, X_mask

    @property
    def data(self):
        return self.X

    def __getitem__(self, index):
        return self.X[0][index], self.X[1][index]

    @property
    def num_instances(self):
        return len(self.X[0])

    @property
    def num_features(self):
        return -1


class _SparseFeatures(SparseFeatures):
    """Class for sparse features
    * Difference: treat 0 as padding index
    
    Arguments
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    X: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the data or not
    """

    def __init__(self, data_dir, fname, X=None, normalize=False):
        super().__init__(data_dir, fname, X, normalize)

    def __getitem__(self, index):
        # Treat idx:0 as Padding
        x = self.X[index].indices + 1
        w = self.X[index].data
        return x, w