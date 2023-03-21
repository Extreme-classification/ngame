import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def clip_batch_lengths(ind, mask, max_len=1000000):
    _max = min(torch.max(torch.sum(mask, dim=1)), max_len)
    return ind[:, :_max], mask[:, :_max]


def pad_and_collate(x, pad_val=0, dtype=torch.FloatTensor):
    """
    A generalized function for padding batch using utils.rnn.pad_sequence
    * pad as per the maximum length in the batch
    * returns a collated tensor

    Arguments:
    ---------
    x: iterator
        iterator over np.ndarray that needs to be converted to
        tensors and padded
    pad_val: float
        pad tensor with this value
        will cast the value as per the data type
    dtype: datatype, optional (default=torch.FloatTensor)
        tensor should be of this type
    """
    return pad_sequence([torch.from_numpy(z) for z in x],
                        batch_first=True, padding_value=pad_val).type(dtype)


def collate_dense(x, dtype=torch.FloatTensor):
    """
    Collate dense documents/labels and returns
    Arguments:
    ---------
    x: iterator
        iterator over np.ndarray that needs to be converted to
        tensors and padded
    dtype: datatype, optional (default=torch.FloatTensor)
        features should be of this type
    """
    return torch.stack([torch.from_numpy(z) for z in x], 0).type(dtype)


def collate_as_1d(x, dtype):
    """
    Collate and return a 1D tensor
    Arguments:
    ---------
    x: iterator
        iterator over np.ndarray that needs to be converted to
        tensors and padded
    dtype: datatype, optional (default=torch.FloatTensor)
        features should be of this type
    """
    return torch.from_numpy(np.concatenate(list(x))).type(dtype)


def collate_as_np_1d(x, dtype):
    """
    Collate and return a 1D tensor
    Arguments:
    ---------
    x: iterator
        iterator over np.ndarray that needs to be converted to
        tensors and padded
    dtype: datatype, optional (default=torch.FloatTensor)
        features should be of this type
    """
    return np.fromiter(x, dtype=dtype)


def collate_sparse(x, pad_val=0.0, has_weight=False, dtype=torch.FloatTensor):
    """
    Collate sparse documents
    * Can handle with or without weights
    * Expects an iterator over tuples if has_weight=True
    Arguments:
    ---------
    x: iterator
        iterator over data points which can be
        np.array or tuple of np.ndarray depending on has_weight
    pad_val: list or float, optional, default=(0.0)
        padding value for indices and weights
        * expects a list when has_weight=True
    has_weight: bool, optional, default=False
        If entries have weights
        * True: objects are tuples of np.ndarrays
            0: indices, 1: weights
        * False: objects are np.ndarrays
    dtypes: list or dtype, optional (default=torch.FloatTensor)
        dtypes of indices and values
        * expects a list when has_weight=True
    """
    weights = None
    if has_weight:
        x = list(x)
        indices = pad_and_collate(map(lambda z: z[0], x), pad_val[0], dtype[0])
        weights = pad_and_collate(map(lambda z: z[1], x), pad_val[1], dtype[1])
    else:
        indices = pad_and_collate(x, pad_val, dtype)
    return indices, weights


def collate_sequential(x):
    # TODO: handle padding
    x = list(x)
    indices = collate_dense(map(lambda z: z[0], x), dtype=torch.LongTensor)
    mask = collate_dense(map(lambda z: z[1], x), dtype=torch.LongTensor)
    return clip_batch_lengths(indices, mask)


def collate_brute(batch):
    return collate_dense(batch), None, None


def collate_implicit_multi_pos(batch):
    batch_labels = []
    unique_labels = set([])
    for item in batch:
        batch_labels.append(item[1])
        unique_labels.update(item[1].tolist())
    unique_labels_l = list(unique_labels)
    label_2_id = {x: i for i, x in enumerate(unique_labels_l)}

    batch_selection =  np.zeros(
        (len(batch_labels), len(unique_labels)), dtype=np.float32)
    for i, item in enumerate(batch_labels):
        intersection = unique_labels.intersection(item)
        result = np.zeros(len(unique_labels))
        for idx in intersection:
            result[label_2_id[idx]] = 1
        batch_selection[i] = result
    
    return torch.from_numpy(batch_selection), \
        torch.LongTensor(unique_labels_l), None


def collate_implicit(batch):
    batch_labels = []
    random_pos_indices = []
    for item in batch:
        random_pos_indices.append(item[0])
        batch_labels.append(item[1])

    batch_size = len(batch_labels)

    batch_selection = np.zeros((batch_size, batch_size), dtype=np.float32)     

    random_pos_indices_set = set(random_pos_indices)
    random_pos_indices = np.array(random_pos_indices, dtype=np.int32)
    

    for (i, item) in enumerate(batch_labels):
        intersection = set(item).intersection(random_pos_indices_set)
        result = np.zeros(batch_size, dtype=np.float32)
        for idx in intersection:
            result += (idx == random_pos_indices)   
        batch_selection[i] = result  

    return torch.from_numpy(batch_selection), \
        torch.LongTensor(random_pos_indices), None


def get_iterator(x, ind=None):
    if ind is None:
        return map(lambda z: z, x)
    else:
        return map(lambda z: z[ind], x)


class collate():
    """
    A generic class to handle different features, classifiers and sampling

    Arguments:
    ----------
    in_feature_t: str
        feature type of input items: dense, sparse or sequential 
    classifier_t: str
        classifier type
        * siamese: label embeddings are treated as classifiers 
        * xc: explicit classifiers 
        * None: no classifiers (used while encoding)
    sampling_t: str
        sampling type
        * implicit: in-batch sampling (use positives of other documents) 
        * explicit: explicit negatives for each labels 
        * brute: 1-vs-all classifiers
    op_feature_t: str
        feature type of output items: dense, sparse, sequential or None 
    """
    def __init__(self, in_feature_t, classifier_t, sampling_t, op_feature_t):
        self.collate_ip_features = self.construct_feature_collator(in_feature_t)
        self.collate_op_features = self.construct_feature_collator(op_feature_t)
        self.collate_labels = self.construct_label_collator(
            classifier_t, sampling_t)

    def construct_feature_collator(self, _type):
        if _type == "dense":
            return collate_dense
        elif _type == "sparse":
            return collate_sparse
        elif _type == "sequential":
            return collate_sequential
        else:
            return None

    def construct_label_collator(self, classifier_t, sampling_t):
        if classifier_t is None:
            return None

        if sampling_t == 'implicit':
            if classifier_t == 'xc':
                return collate_implicit_multi_pos
            else:
                return collate_implicit
        elif sampling_t == 'explicit':
            raise NotImplementedError("")
        elif sampling_t == 'brute':
            return collate_brute
        else:
            raise NotImplementedError("")

    def __call__(self, batch):
        data = {}
        data['batch_size'] = torch.tensor(len(batch), dtype=torch.int32)
        data['X'] = self.collate_ip_features(get_iterator(batch, 0))
        data['indices'] = torch.LongTensor([item[-1] for item in batch])
        if self.collate_labels is not None: # labels are availabels
            data['Y'], data['Y_s'], data['Y_mask'] = self.collate_labels(
                get_iterator(batch, 1))
        if self.collate_op_features is not None: # label features are available
            data['Z'] = self.collate_op_features(get_iterator(batch, 2))
        return data
