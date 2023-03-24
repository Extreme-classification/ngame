import torch
import pickle
import os
import numpy as np
from .features import construct as construct_f
from .labels import construct as construct_l


class DatasetBase(torch.utils.data.Dataset):
    """Dataset to load and use XML-Datasets

    Support pickle or libsvm file format
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
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
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    """

    def __init__(self, data_dir,
                 f_features,
                 f_labels,
                 data={'X': None, 'Y': None, 'Yf': None},
                 model_dir='',
                 mode='train',
                 normalize_features=True,
                 normalize_lables=False,
                 feature_type='sparse',
                 label_type='dense',
                 f_label_features=None,
                 max_len=-1,
                 *args, **kwargs):
        if data is None:
            data = {'X': None, 'Y': None, 'Yf': None}
        self.mode = mode
        self.features, self.labels, self.label_features = self.load_data(
            data_dir, f_features, f_labels,
            data, normalize_features, normalize_lables,
            feature_type, label_type, f_label_features, max_len)
        self.model_dir = model_dir
        self.label_padding_index = self.num_labels

    def _remove_samples_wo_features_and_labels(self):
        """Remove instances if they don't have any feature or label
        """
        indices = self.features.get_valid_indices(axis=1)
        if self.labels is not None:
            indices_labels = self.labels.get_valid_indices(axis=1)
            indices = np.intersect1d(indices, indices_labels)
            self.labels._index_select(indices, axis=0)
        self.features._index_select(indices, axis=0)

    def index_select(self, feature_indices, label_indices):
        """Transform feature and label matrix to specified
        features/labels only
        """
        if label_indices is not None:
            label_indices = np.loadtxt(label_indices, dtype=np.int32)
            self.labels._index_select(label_indices, axis=1)
            if self.label_features is not None:
                self.label_features._index_select(label_indices, axis=0)
        if feature_indices is not None:
            feature_indices = np.loadtxt(feature_indices, dtype=np.int32)
            self.features._index_select(feature_indices, axis=1)
            if self.label_features is not None:
                self.label_features._index_select(feature_indices, axis=1)

    def load_features(self, data_dir, fname, X,
                      normalize_features, feature_type, max_len):
        """Load features from given file
        Features can also be supplied directly
        """
        return construct_f(data_dir, fname, X,
                           normalize_features,
                           feature_type, max_len)

    def load_labels(self, data_dir, fname, Y, normalize_labels, label_type):
        """Load labels from given file
        Labels can also be supplied directly
        """
        labels = construct_l(data_dir, fname, Y, normalize_labels,
                             label_type)  # Pass dummy labels if required
        if normalize_labels:
            if self.mode == 'train':  # Handle non-binary labels
                print("Non-binary labels encountered in train; Normalizing.")
                labels.normalize(norm='max', copy=False)
            else:
                print("Non-binary labels encountered in test/val; Binarizing.")
                labels.binarize()
        return labels

    def load_data(self, data_dir, f_features, f_labels, data,
                  normalize_features=True, normalize_labels=False,
                  feature_type='sparse', label_type='dense',
                  f_label_features=None, max_len=32):
        """Load features and labels from file in libsvm format or pickle
        """
        features = self.load_features(
            data_dir, f_features, data['X'],
            normalize_features, feature_type, max_len)
        labels = self.load_labels(
            data_dir, f_labels, data['Y'], normalize_labels, label_type)
        label_features = None
        if f_label_features is not None or data["Yf"] is not None:
            label_features = self.load_features(
                data_dir, f_label_features, data['Yf'],
                normalize_features, feature_type, max_len)
        return features, labels, label_features

    @property
    def num_instances(self):
        return self.features.num_instances

    @property
    def num_features(self):
        return self.features.num_features

    @property
    def num_labels(self):
        return self.labels.num_labels

    def get_stats(self):
        """Get dataset statistics
        """
        return self.num_instances, self.num_features, self.num_labels

    def _process_labels_train(self, data_obj):
        """Process labels for train data
            - Remove labels without any training instance
        #TODO: Handle label features
        """
        data_obj['num_labels'] = self.num_labels
        valid_labels = self.labels.remove_invalid()
        data_obj['valid_labels'] = valid_labels

    def _process_labels_predict(self, data_obj):
        """Process labels for test data
           Only use valid labels i.e. which had atleast one training
           example
        #TODO: Handle label features
        """
        valid_labels = data_obj['valid_labels']
        self.labels._index_select(valid_labels, axis=1)

    def _process_labels(self, model_dir, _split=None):
        """Process labels to handle labels without any training instance;
        """
        data_obj = {}
        fname = os.path.join(
            model_dir, 'labels_params.pkl' if _split is None else
            "labels_params_split_{}.pkl".format(_split))
        if self.mode == 'train':
            self._process_labels_train(data_obj)
            pickle.dump(data_obj, open(fname, 'wb'))
        else:
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_predict(data_obj)

    def __len__(self):
        return self.num_instances

    @property
    def feature_type(self):
        return self.features._type

    def __getitem__(self, index):
        """Get features and labels for index
        Arguments
        ---------
        index: int
            data for this index
        """
        raise NotImplementedError("")


class DatasetSampling(DatasetBase):
    """Dataset to load and use XML-Datasets
    with shortlist
    
    Support pickle or libsvm file format
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
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
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    """

    def __init__(self,
                 data_dir,
                 f_features,
                 f_labels,
                 f_label_features,
                 data=None,
                 model_dir='',
                 mode='train',
                 normalize_features=True,
                 sampling_params=None,
                 normalize_lables=False,
                 feature_type='sparse',
                 max_len=-1,
                 label_type='dense'):
        super().__init__(
            data_dir=data_dir,
            f_features=f_features,
            f_labels=f_labels,
            f_label_features=f_label_features,
            data=data,
            model_dir=model_dir,
            mode=mode, 
            normalize_features=normalize_features,
            normalize_lables=normalize_lables,
            feature_type=feature_type,
            max_len=max_len,
            label_type=label_type)
        self.sampler = self.construct_sampler(sampling_params)

    def construct_sampler(self, sampling_params):
        return None

    def indices_permutation(self):
        return np.arange(len(self.features))


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

    def __init__(self,
                 data_dir,
                 f_features,
                 data=None,
                 feature_indices=None,
                 normalize_features=True,
                 feature_type='sparse',
                 **kwargs):
        self.data = self.construct(
            data_dir, f_features, data, feature_indices,
            normalize_features, feature_type, **kwargs)

    def construct(self, data_dir, fname, data, indices,
                  normalize, _type, **kwargs):
        data = construct_f(data_dir, fname, data, normalize, _type, **kwargs)
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
        return (self.data[index], index)
