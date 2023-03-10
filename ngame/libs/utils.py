import os
from scipy.sparse import save_npz
from xclib.utils.sparse import _map_cols


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