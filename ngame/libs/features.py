from xclib.data.features import SparseFeatures
from xclib.data.features import DenseFeatures as _DenseFeatures
from xclib.data import data_utils
import os
import numpy as np


def construct(data_dir, fname, X=None, normalize=False,
              _type='sparse', max_len=-1, **kwargs):
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
    max_len: int, optional, default=-1
        max length in sequential features
    """
    if _type == 'sparse':
        return _SparseFeatures(data_dir, fname, X, normalize)
    elif _type == 'dense':
        return DenseFeatures(data_dir, fname, X, normalize)
    elif _type == 'sequential':
        return SeqFeatures(data_dir, fname, X, max_len)
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
    max_len: 32, optional, default=32
        max length in sequential features
    """
    def __init__(self, data_dir, fname, X=None, max_len=-1):
        super().__init__(data_dir, fname, X)
        self.max_len = max_len

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
        if self.max_len > 0:
            return self.X[0][index][:self.max_len], \
                self.X[1][index][:self.max_len]
        else:
            return (self.X[0][index], self.X[1][index])

    @property
    def num_instances(self):
        return len(self.X[0])

    @property
    def num_features(self):
        return -1

    @property
    def _type(self):
        return 'sequential'

    @property
    def _params(self):
        return {'max_len': self.max_len,
                'feature_type': self._type,
                '_type': self._type}


class DenseFeatures(_DenseFeatures):
    @property
    def _type(self):
        return 'dense'

    @property
    def _params(self):
        return {'feature_type': self._type,
                '_type': self._type}


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

    @property
    def _type(self):
        return 'sparse'

    @property
    def _params(self):
        return {'feature_type': self._type,
                '_type': self._type}
