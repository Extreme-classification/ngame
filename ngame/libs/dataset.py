import torch
from xclib.data import data_utils
import os
import numpy as np
from .features import SeqFeatures
from .dataset_base import DatasetTensor, DatasetBase
from .shortlist import ClusteringIndex


class DatasetOVA(DatasetBase):
    """Dataset to load and use XML-Datasets with full output space only
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm/pickle/numpy)
    fname_labels: str
        labels file (libsvm/pickle/npz)    
    fname_label_features: str
        feature file for labels (libsvm/pickle/npy)
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y', 'Yf'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    feature_type: str, optional, default='sparse'
        sparse or dense features
    """

    def __init__(self,
                 data_dir,
                 fname_features,
                 fname_labels,
                 fname_label_features=None,
                 data=None,
                 model_dir='',
                 mode='train', 
                 normalize_features=True,
                 normalize_labels=False,
                 feature_type='sparse'):
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, normalize_features, normalize_labels,
                         feature_type, label_type='dense')
        self.label_features = self.load_features(
            data_dir, fname_label_features, data['Yf'],
            normalize_features, feature_type)
        self.feature_type = feature_type
        # TODO Take care of this select and padding index
        self.label_padding_index = self.num_labels

    def __getitem__(self, index):
        """
            Get features and labels for index
            Args:
                index: for this sample
            Returns:
                features: : non zero entries
                labels: : numpy array
        """
        x = self.features[index]
        y = self.labels[index]
        return x, y 


class DatasetIB(DatasetBase):
    def __init__(self,
                 data_dir,
                 fname_features,
                 fname_labels,
                 fname_label_features=None,
                 data=None,
                 model_dir='',
                 mode='train', 
                 normalize_features=True,
                 normalize_labels=False,
                 feature_type='sparse'):
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, normalize_features, normalize_labels,
                         feature_type, label_type='dense')
        self.label_features = self.load_features(
            data_dir, fname_label_features, data['Yf'],
            normalize_features, feature_type)
        self.feature_type = feature_type
        # TODO Take care of this select and padding index
        self.label_padding_index = self.num_labels

    def __getitem__(self, index):
        """
            Get features and labels for index
            Args:
                index: for this sample
            Returns:
                features: : non zero entries
                labels: : numpy array
        """
        ind, mask = self.features[index]
        pos_indices = self.labels[index].indices
        sampled_pos_ind = np.random.choice(pos_indices)
        l_ind, l_mask = self.lbl_features[sampled_pos_ind]
        return (ind, mask, pos_indices, sampled_pos_ind, l_ind, l_mask, index)



        x = self.features[index]
        y = self.labels[index]
        return x, y 


class DatasetSL():
    pass


class SDatasetIB():
    pass


class SDatasetSL():
    pass

class XDatasetIB():
    pass


class XDatasetSL():
    pass

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, feature_fname,
                 label_feature_fname, label_fname,
                 sampling_params, max_len):
        print(data_dir, feature_fname, label_feature_fname, label_fname, max_len)
        self.max_len = max_len
        #self.feature_type = "seq"
        self.labels = data_utils.read_sparse_file(os.path.join(data_dir, label_fname))
        self._valid_labels = np.arange(self.labels.shape[1])
        self.features = SeqFeatures(data_dir, feature_fname)
        self.lbl_features = SeqFeatures(data_dir, label_feature_fname)
        self.shortlist = self.construct_shortlist_handler(sampling_params)

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

    def construct_shortlist_handler(self, sampling_params):
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


class SDatasetBDIS(Dataset):
    def __init__(self, data_dir, feature_fname, label_feature_fname, label_fname, sampling_params, max_len):
        super().__init__(data_dir, feature_fname, label_feature_fname, label_fname, sampling_params, max_len)

    def construct_shortlist_handler(self, sampling_params):
        if sampling_params is not None:
            return ClusteringIndex(
                num_instances=self.__len__(),
                num_clusters=self.__len__(),
                num_threads=sampling_params.sampling_threads,
                curr_steps=sampling_params.sampling_curr_epochs)

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

    def update_state(self, *args):
        self.shortlist.update_state()

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



class XDatasetBDIS(Dataset):
    def __init__(self, data_dir, feature_fname, label_feature_fname, label_fname, sampling_params, max_len):
        super().__init__(data_dir, feature_fname, label_feature_fname, label_fname, sampling_params, max_len)

    def construct_shortlist_handler(self, sampling_params):
        if sampling_params is not None:
            return ClusteringIndex(
                num_instances=self.__len__(),
                num_clusters=self.__len__(),
                num_threads=sampling_params.sampling_threads,
                curr_steps=sampling_params.sampling_curr_epochs)

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

    def update_state(self, *args):
        self.shortlist.update_state()

    def update_shortlist(self, *args):
        """Update label shortlist for given document index
        """
        self.shortlist.update(*args)

    def __getitem__(self, index):
        """Get a label at index"""
        ind, mask = self.features[index]
        pos_indices = self.labels[index].indices
        sampled_pos_ind = np.random.choice(pos_indices)
        return (ind, mask, pos_indices, sampled_pos_ind, index)


class SDatasetBLIS(Dataset):
    def __init__(self, data_dir, feature_fname, label_feature_fname, label_fname, sampling_params, max_len):
        super().__init__(data_dir, feature_fname, label_feature_fname, label_fname, max_len)
        self._labels = self.labels.T.tocsr()

    def construct_shortlist_handler(self, sampling_params):
        if sampling_params is not None:
            return ClusteringIndex(
                num_instances=self.__len__(),
                num_clusters=self.__len__(),
                num_threads=sampling_params.sampling_threads,
                curr_steps=sampling_params.sampling_curr_epochs)

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

    def update_state(self, *args):
        self.shortlist.update_state()

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

