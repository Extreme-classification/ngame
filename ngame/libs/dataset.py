import torch
from xclib.data import data_utils
import os
import numpy as np
from .features import SeqFeatures
from .dataset_base import DatasetTensor
from .shortlist import ClusteringIndex


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, feature_fname,
                 label_feature_fname, label_fname,
                 max_len, shortlist_method=None,
                 size_shortlist=-1, n_threads=6):
        print(data_dir, feature_fname, label_feature_fname, label_fname, max_len)
        self.max_len = max_len
        #self.feature_type = "seq"
        self.labels = data_utils.read_sparse_file(os.path.join(data_dir, label_fname))
        self._valid_labels = np.arange(self.labels.shape[1])
        self.features = SeqFeatures(data_dir, feature_fname)
        self.lbl_features = SeqFeatures(data_dir, label_feature_fname)
        self.shortlist = self.construct_shortlist_handler(
            shortlist_method, size_shortlist, n_threads)

    def indices_permutation(self):
        return np.random.permutation(self.num_instances)

    def get_shortlist(self, index):
        """Get label shortlist for given document index
        """
        pass

    def update_shortlist(self, index):
        """Update label shortlist for given document index
        """
        pass

    def construct_shortlist_handler(self, shortlist_method, size_shortlist, n_threads):
        return None

    def __getitem__(self, index):
        raise NotImplementedError("")

    @property
    def num_instances(self):
        return self.labels.shape[0]

    @property
    def num_labels(self):
        return self.labels.shape[1]

    def __len__(self):
        return self.num_instances

    @property
    def feature_type(self):
        return 'sequential'


class DatasetBDIS(Dataset):
    def __init__(self, data_dir, feature_fname, label_feature_fname, label_fname, max_len):
        super().__init__(data_dir, feature_fname, label_feature_fname, label_fname, max_len)

    def construct_shortlist_handler(self, shortlist_method, size_shortlist, n_threads):
        return ClusteringIndex(self.__len__(), self.__len__(), n_threads)

    def indices_permutation(self):
        clusters = self.shortlist.index
        np.random.shuffle(clusters)
        indices = []
        for item in clusters:
            indices.extend(item)
        return np.array(indices)

    def get_shortlist(self, index):
        """Get label shortlist for given document index
        """
        return self.shortlist.query(index)

    def update_shortlist(self, *args):
        """Update label shortlist for given document index
        """
        self.shortlist.update(*args)

    def __getitem__(self, index):
        """Get a label at index"""
        ind, mask = self.features[index]
        pos_indices = self.labels[index].indices
        sampled_pos_ind = np.random.choice(pos_indices)
        l_ind, l_mask = self.lbl_features[sampled_pos_ind]
        return (ind, mask, pos_indices, sampled_pos_ind, l_ind, l_mask, index)

    # def __getitem__(self, index):
    #     """Get a label at index"""
    #     idx = self.get_shortlist(index)
    #     ind, mask = self.features[idx]
    #     pos_indices = [self.labels[i].indices for i in idx]
    #     sampled_pos_ind = [np.random.choice(p) for p in pos_indices] #np.random.choice(pos_indices)
    #     l_ind, l_mask = self.lbl_features[sampled_pos_ind]
    #     return (ind, mask, pos_indices, sampled_pos_ind, l_ind, l_mask, index)


class DatasetBLIS(Dataset):
    def __init__(self, data_dir, feature_fname, label_feature_fname, label_fname, max_len):
        super().__init__(data_dir, feature_fname, label_feature_fname, label_fname, max_len)
        self._labels = self.labels.T.tocsr()

    def construct_shortlist_handler(self, shortlist_method, size_shortlist, n_threads):
        return ClusteringIndex(self.__len__(), self.__len__(), n_threads)

    def indices_permutation(self):
        clusters = self.shortlist.index
        np.random.shuffle(clusters)
        indices = []
        for item in clusters:
            indices.extend(item)
        return indices

    def get_shortlist(self, index):
        """Get document shortlist for given label index
        """
        return self.shortlist.query(index)

    def update_shortlist(self, *args):
        """Update document shortlist for given label index
        """
        self.shortlist.update(*args)

    # def __getitem__(self, index):
    #     """Get a label at index"""
    #     idx = self.get_shortlist(index)
    #     ind, mask = self.lbl_features[idx]
    #     pos_indices = [self._labels[i].indices for i in idx]
    #     sampled_pos_ind = [np.random.choice(p) for p in pos_indices] #np.random.choice(pos_indices)
    #     l_ind, l_mask = self.features[sampled_pos_ind]
    #     return (ind, mask, pos_indices, sampled_pos_ind, l_ind, l_mask, index)

    def __getitem__(self, index):
        """Get a label at index"""
        ind, mask = self.lbl_features[index]
        pos_indices = self._labels[index].indices
        sampled_pos_ind = np.random.choice(pos_indices)
        d_ind, d_mask = self.features[sampled_pos_ind]
        return (ind, mask, pos_indices, sampled_pos_ind, d_ind, d_mask, index)

    def __len__(self):
        return self.num_labels

