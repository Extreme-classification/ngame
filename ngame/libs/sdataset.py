import numpy as np
from .shortlist import ClusteringIndex
from libs.dataset_base import DatasetSampling


class DatasetBDIS(DatasetSampling):
    def __init__(self,
                 data_dir,
                 f_features,
                 f_label_features,
                 f_labels,
                 sampling_params,
                 max_len,
                 normalize_features=True,
                 normalize_lables=False,
                 feature_type='sparse',
                 label_type='dense',
                 data=None,
                 model_dir='',
                 mode='predict',
                 *args,
                 **kwargs
                ):
        super().__init__(data_dir=data_dir,
                         f_features=f_features,
                         data=data,
                         f_label_features=f_label_features,
                         f_labels=f_labels,
                         sampling_params=sampling_params,
                         max_len=max_len,
                         normalize_features=normalize_features,
                         normalize_lables=normalize_lables,
                         feature_type=feature_type,
                         label_type=label_type,
                         mode=mode,
                         model_dir=model_dir
                        )

    def construct_sampler(self, sampling_params):
        if sampling_params is not None:
            return ClusteringIndex(
                num_instances=self.__len__(),
                num_clusters=self.__len__(),
                num_threads=sampling_params.threads,
                curr_steps=sampling_params.curr_epochs)

    def indices_permutation(self):
        clusters = self.sampler.index
        np.random.shuffle(clusters)
        indices = []
        for item in clusters:
            indices.extend(item)
        return np.array(indices)

    def get_sampler(self, index):
        """Get negatives for a given index
        """
        return self.sampler.query(index)

    def update_state(self, *args):
        self.sampler.update_state()

    def update_sampler(self, *args):
        """Update negative sampler
        """
        self.sampler.update(*args)

    def __getitem__(self, index):
        """Get a label at index"""
        doc_ft = self.features[index]
        pos_indices, _ = self.labels[index]
        sampled_pos_ind = np.random.choice(pos_indices)
        lbl_ft = self.label_features[sampled_pos_ind]
        return (doc_ft, (sampled_pos_ind, pos_indices), lbl_ft, index) 


class DatasetBLIS(DatasetSampling):
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

    def __getitem__(self, index):
        """Get a label at index"""
        ind, mask = self.lbl_features[index]
        pos_indices = self._labels[index].indices
        sampled_pos_ind = np.random.choice(pos_indices)
        d_ind, d_mask = self.features[sampled_pos_ind]
        return (ind, mask, pos_indices, sampled_pos_ind, d_ind, d_mask, index)

    def __len__(self):
        return self.num_labels
 
