import os
from scipy.sparse import save_npz
from xclib.utils.sparse import _map_cols, csr_from_arrays
import numpy as np
from .shortlist import ShortlistMIPS
from argparse import Namespace


def save_predictions(preds, result_dir, valid_labels, num_samples,
                     num_labels, get_fnames=['knn', 'clf', 'combined'],
                     prefix='predictions'):
    if isinstance(preds, dict):
        for _fname, _pred in preds.items():
            if _fname in get_fnames:
                if valid_labels is not None:
                    predicted_labels = _map_cols(
                        _pred, valid_labels, shape=(num_samples, num_labels))
                else:
                    predicted_labels = _pred
                save_npz(os.path.join(
                    result_dir, '{}_{}.npz'.format(prefix, _fname)),
                    predicted_labels, compressed=False)
    else:
        if valid_labels is not None:
            predicted_labels = _map_cols(
                preds, valid_labels, shape=(num_samples, num_labels))
        else:
            predicted_labels = preds
        save_npz(os.path.join(result_dir, '{}.npz'.format(prefix)),
                 predicted_labels, compressed=False)
        

def get_filter_map(fname):
    if fname is not None:
        return np.loadtxt(fname).astype(np.int)
    else:
        return None


def filter_predictions(pred, mapping):
    if mapping is not None and len(mapping) > 0:
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def map_filter_mapping(mapping, valid_labels, num_labels):
    if mapping is None or valid_labels is None:
        return mapping
    val_ind = []
    valid_mapping = dict(
        zip(valid_labels, np.arange(num_labels)))
    for i in range(len(mapping)):
        if mapping[i, 1] in valid_mapping:
            mapping[i, 1] = valid_mapping[mapping[i, 1]]
            val_ind.append(i)
    return mapping[val_ind]


def predict_anns(X, W, k=300, method='hnswlib', space='cosine',
                M=100, efC=300, n_threads=6, add_padding=False):
    """
    Train a nearest neighbor structure on W
    - for a given test point: query the graph for closest label
    """
    num_instances, num_labels = len(X), len(W)
    
    # add a padding index in the end
    if add_padding:
        num_labels += 1
    
    # can handle zero vectors
    graph = ShortlistMIPS(
        method=method,
        M=M,
        efC=efC,
        efS=k,
        num_neighbours=k,
        space=space, 
        num_threads=n_threads)    
    graph.fit(W)
    ind, sim = graph.query(X)
    pred = csr_from_arrays(ind, sim, (num_instances, num_labels))
    return pred


def filter_params(args, prefix):
    """
    Filter the arguments as per a prefix from a given namespace
    """
    out = {}
    for k, v in args.__dict__.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return Namespace(**out)


def load_token_emeddings(data_dir, embeddings, feature_indices=None):
    """Load word embeddings from numpy file
    * Support for:
        - loading pre-trained embeddings
        - generating random embeddings
    * vocabulary_dims must match #rows in embeddings
    """
    try:
        embeddings = np.load(os.path.join(data_dir, embeddings))
    except FileNotFoundError:
        exit("Embedding File not found. Check path or set 'init' to null")
    if feature_indices is not None:
        indices = np.genfromtxt(feature_indices, dtype=np.int32)
        embeddings = embeddings[indices, :]
        del indices
    return embeddings