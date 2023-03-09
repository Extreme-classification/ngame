import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


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


def construct_selection(sel_pos_indices, pos_indices):
    # Will use numpy; pytorch intersect1d is weird
    batch_size = pos_indices.shape[0]
    selection = np.zeros((batch_size, batch_size), dtype=np.float32)

    sel_pos_indices_set = set(sel_pos_indices)
    for (i, item) in enumerate(pos_indices):
        intersection = set(item).intersection(sel_pos_indices_set)
        result = np.zeros(batch_size, dtype=np.float32)
        for idx in intersection:
            result += (idx == sel_pos_indices)
        selection[i] = result
    return torch.from_numpy(selection)


def get_iterator(x, ind=None):
    if ind is None:
        return map(lambda z: z, x)
    else:
        return map(lambda z: z[ind], x)


def construct_collate_fn(feature_type, classifier_type='dense'):
    def _collate_fn_dense_full(batch):
        return collate_fn_dense_full(batch)

    def _collate_fn_dense(batch):
        return collate_fn_dense(batch)

    def _collate_fn_sparse(batch):
        return collate_fn_sparse(batch)

    def _collate_fn_dense_sl(batch):
        return collate_fn_dense_sl(batch)

    def _collate_fn_sparse_embedding(batch):
        return collate_fn_sparse_embedding(batch)

    def _collate_fn_sparse_sl(batch):
        return collate_fn_sparse_sl(batch)

    if feature_type == 'dense':
        if classifier_type == 'None':
            return _collate_fn_dense
        elif classifier_type == 'shortlist':
            return _collate_fn_dense_sl
        else:
            return _collate_fn_dense_full
    elif feature_type == 'sparse':
        if classifier_type == 'None':
            return _collate_fn_sparse
        elif classifier_type == 'shortlist':
            return _collate_fn_sparse_sl
        else:
            return _collate_fn_sparse_embedding
    elif feature_type == 'sequential':
        if classifier_type == 'None':
            return _collate_fn_seq
        elif classifier_type == 'shortlist':
            return _collate_fn_seq_sl
        else:
            return _collate_fn_seq_embedding


def collate_fn_sparse_sl(batch):
    """
        Combine each sample in a batch with shortlist
        For sparse features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X_ind'], batch_data['X'] = collate_sparse(
        get_iterator(batch, 0), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])

    z = list(get_iterator(batch, 1))
    batch_data['Y_s'] = collate_dense(
        get_iterator(z, 0), dtype=torch.LongTensor)
    batch_data['Y'] = collate_dense(
        get_iterator(z, 1), dtype=torch.FloatTensor)
    batch_data['Y_sim'] = collate_dense(
        get_iterator(z, 2), dtype=torch.FloatTensor)
    batch_data['Y_mask'] = collate_dense(
        get_iterator(z, 3), dtype=torch.BoolTensor)
    return batch_data


def collate_fn_dense_sl(batch):
    """
        Combine each sample in a batch with shortlist
        For dense features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X'] = collate_dense(get_iterator(batch, 0))

    z = list(get_iterator(batch, 1))
    batch_data['Y_s'] = collate_dense(
        get_iterator(z, 0), dtype=torch.LongTensor)
    batch_data['Y'] = collate_dense(
        get_iterator(z, 1), dtype=torch.FloatTensor)
    batch_data['Y_sim'] = collate_dense(
        get_iterator(z, 2), dtype=torch.FloatTensor)
    batch_data['Y_mask'] = collate_dense(
        get_iterator(z, 3), dtype=torch.BoolTensor)
    return batch_data


def collate_fn_dense_full(batch):
    """
        Combine each sample in a batch
        For dense features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X'] = collate_dense(get_iterator(batch, 0))
    batch_data['Y'] = collate_dense(get_iterator(batch, 1))
    return batch_data


def collate_fn_sparse_embedding(batch):
    """
        Combine each sample in a batch
        For sparse features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X_ind'], batch_data['X'] = collate_sparse(
        get_iterator(batch, 0), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])
    sel_pos_indices = collate_as_np_1d(
        get_iterator(batch, 1), 'int')
    pos_indices, _ = collate_sparse(
        get_iterator(batch, 4), pad_val=-1.0)
    batch_data['Y'] = construct_selection(
        sel_pos_indices, pos_indices.numpy().astype('int'))
    batch_data['YX_ind'], batch_data['YX'] = collate_sparse(
        get_iterator(batch, 2), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])
    batch_data['CX_ind'], batch_data['CX'] = collate_sparse(
        get_iterator(batch, 3), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])
    batch_data['Y_mask'] = None
    return batch_data


def collate_fn_dense_embedding(batch):
    """
        Combine each sample in a batch
        For sparse features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X_ind'], batch_data['X'] = collate_sparse(
        get_iterator(batch, 0), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])
    batch_data['Y'] = collate_dense(get_iterator(batch, 1))
    batch_data['Y_mask'] = None
    return batch_data


def collate_fn_sparse(batch):
    """
        Combine each sample in a batch
        For sparse features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X_ind'], batch_data['X'] = collate_sparse(
        get_iterator(batch), pad_val=[0, 0.0], has_weight=True,
        dtype=[torch.LongTensor, torch.FloatTensor])
    return batch_data


def collate_fn_dense(batch):
    """
        Combine each sample in a batch
        For sparse features
    """
    batch_data = {'batch_size': len(batch), 'X_ind': None}
    batch_data['X'] = collate_dense(get_iterator(batch))
    return batch_data


def _collate_fn_seq(batch):
    batch_data = {}
    batch_data['batch_size'] = len(batch)
    batch_data['X'] = None
    batch_data['ind'] = torch.from_numpy(np.vstack([item[0] for item in batch]))
    batch_data['mask'] = torch.from_numpy(np.vstack([item[1] for item in batch]))
    return batch_data


def _collate_fn_ext(batch, max_len):
    batch_labels = []
    random_pos_indices = []
    for item in batch:
        batch_labels.append(item[2])
        random_pos_indices.append(item[3])

    batch_size = len(batch_labels)

    ind = np.vstack([x[0] for x in batch]) 
    mask = np.vstack([x[1] for x in batch])
    batch_selection = np.zeros((batch_size, batch_size), dtype=np.float32)     

    random_pos_indices_set = set(random_pos_indices)
    random_pos_indices = np.array(random_pos_indices, dtype=np.int32)
    

    for (i, item) in enumerate(batch_labels):
        intersection = set(item).intersection(random_pos_indices_set)
        result = np.zeros(batch_size, dtype=np.float32)
        for idx in intersection:
            result += (idx == random_pos_indices)   
        batch_selection[i] = result  

    return ind, mask, random_pos_indices, batch_selection


def _collate_fn(batch, max_len):
    batch_labels = []
    random_pos_indices = []
    for item in batch:
        batch_labels.append(item[2])
        random_pos_indices.append(item[3])

    batch_size = len(batch_labels)

    ip_ind = np.vstack([x[0] for x in batch]) 
    ip_mask = np.vstack([x[1] for x in batch])
    op_ind = np.vstack([x[4] for x in batch])
    op_mask = np.vstack([x[5] for x in batch])
    batch_selection = np.zeros((batch_size, batch_size), dtype=np.float32)     

    random_pos_indices_set = set(random_pos_indices)
    random_pos_indices = np.array(random_pos_indices, dtype=np.int32)
    

    for (i, item) in enumerate(batch_labels):
        intersection = set(item).intersection(random_pos_indices_set)
        result = np.zeros(batch_size, dtype=np.float32)
        for idx in intersection:
            result += (idx == random_pos_indices)   
        batch_selection[i] = result  

    return ip_ind, ip_mask, op_ind, op_mask, batch_selection

# def _collate_fn(batch, max_len):
#     batch_size = len(batch)
#     _max = -1
#     for item in batch:
#         _max = max(_max, len(item[2]))

#     ip_ind = np.vstack([x[0] for x in batch]) #np.zeros((batch_size, max_len), dtype=np.int64)
#     ip_mask = np.zeros((batch_size, max_len), dtype=np.int64)
#     op_ind = np.zeros((batch_size, max_len), dtype=np.int64)
#     op_mask = np.zeros((batch_size, max_len), dtype=np.int64)
#     batch_selection = np.zeros((batch_size, batch_size), dtype=np.float32)     
#     random_pos_indices = np.zeros(batch_size, dtype=np.int32)
    
#     for (i, item) in enumerate(batch):
#         ip_ind[i] = item[0]
#         ip_mask[i] = item[1]
         
#         random_pos_indices[i] = item[3]
        
#         op_ind[i] = item[4]
#         op_mask[i] = item[5]

#     random_pos_indices_set = set(random_pos_indices)
#     for (i, item) in enumerate(batch):
#         intersection = set(item[2]).intersection(random_pos_indices_set)
#         result = np.zeros(batch_size, dtype=np.float32)
#         for idx in intersection:
#             result += (idx == random_pos_indices)
   
#         batch_selection[i] = result  

#     return ip_ind, ip_mask, op_ind, op_mask, batch_selection


def clip_batch_lengths(ind, mask):
    _max = np.max(np.sum(mask, axis=1))
    return ind[:, :_max], mask[:, :_max]


def collate_fn_seq_siamese(batch, max_len=32):
    batch_data = {}
    batch_size = len(batch)
    batch_data['batch_size'] = torch.tensor(batch_size, dtype=torch.int32)
    
    ip_ind, ip_mask, op_ind, op_mask, batch_selection = _collate_fn(batch, max_len)
    ip_ind, ip_mask = clip_batch_lengths(ip_ind, ip_mask)
    op_ind, op_mask = clip_batch_lengths(op_ind, op_mask)

    batch_data['indices'] = torch.LongTensor([item[-1] for item in batch])
    batch_data['ip_ind'] = torch.from_numpy(ip_ind)
    batch_data['ip_mask'] = torch.from_numpy(ip_mask)
    batch_data['op_ind'] = torch.from_numpy(op_ind)
    batch_data['op_mask'] = torch.from_numpy(op_mask)
    batch_data['Y'] = torch.from_numpy(batch_selection)
    batch_data['Y_mask'] = None
    
    return batch_data


def collate_fn_seq_xc(batch, max_len=32):
    batch_data = {}
    batch_size = len(batch)
    batch_data['batch_size'] = torch.tensor(batch_size, dtype=torch.int32)
    
    ind, mask, lbl_indices, batch_selection = _collate_fn_ext(batch, max_len)
    ind, mask = clip_batch_lengths(ind, mask)

    batch_data['indices'] = torch.LongTensor([item[-1] for item in batch])
    batch_data['ind'] = torch.from_numpy(ind)
    batch_data['mask'] = torch.from_numpy(mask)
    batch_data['Y_s'] = torch.LongTensor(lbl_indices)
    batch_data['Y'] = torch.from_numpy(batch_selection)
    batch_data['Y_mask'] = None
    
    return batch_data
