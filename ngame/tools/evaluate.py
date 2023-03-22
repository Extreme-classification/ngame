# Example to evaluate
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.sparse import load_npz, save_npz
from xclib.utils.sparse import retain_topk
import numpy as np
import os
from tools.score_fusion import ScoreFusion


def get_filter_map(fname):
    if fname is not None:
        return np.loadtxt(fname).astype(np.int)
    else:
        return None


def filter_predictions(pred, mapping):
    if mapping is not None and len(mapping) > 0:
        print("Filtering labels.")
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def eval(tst_label_fname, trn_label_fname, pred_fname,
         A, B, save, filter_fname=None, top_k=200, *args, **kwargs):
    ans = ""
    true_labels = data_utils.read_sparse_file(tst_label_fname)
    trn_labels = data_utils.read_sparse_file(trn_label_fname)
    inv_psp = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    mapping = get_filter_map(filter_fname)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_psp)
    root = os.path.dirname(pred_fname)

    pred = filter_predictions(load_npz(pred_fname), mapping)
 

    args = acc.eval(pred.copy(), 5)
    ans = f"classifier\n{xc_metrics.format(*args)}"

    if save:
        fname = os.path.join(root, f"score.npz")
        save_npz(fname, retain_topk(pred, k=top_k),
            compressed=False)
    return ans


def eval_with_score_fusion(tst_label_fname, trn_label_fname, trn_pred_fname,
                           pred_fname, A, B, beta, save,
                           filter_fname=None, trn_filter_fname=None, top_k=200):
    ans = ""
    true_labels = data_utils.read_sparse_file(tst_label_fname)
    trn_labels = data_utils.read_sparse_file(trn_label_fname)
    inv_psp = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    mapping = get_filter_map(filter_fname)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_psp)
    root = os.path.dirname(pred_fname)

    knn = filter_predictions(
        load_npz(pred_fname+'_knn.npz'), mapping)
    clf = filter_predictions(
        load_npz(pred_fname+'_clf.npz'), mapping)
    args = acc.eval(clf, 5)
    ans = f"classifier\n{xc_metrics.format(*args)}"
    args = acc.eval(knn, 5)
    ans = ans + f"\nshortlist\n{xc_metrics.format(*args)}"

    trn_knn = filter_predictions(
        load_npz(trn_pred_fname+'_knn.npz'), mapping)
    trn_clf = filter_predictions(
        load_npz(trn_pred_fname+'_clf.npz'), mapping)

    f = ScoreFusion(inv_psp=inv_psp)
    f.fit(
        val_pred_clf=trn_clf,
        val_pred_knn=trn_knn,
        val_lbl=filter_predictions(
            trn_labels.copy(), get_filter_map(trn_filter_fname)),
        ns=75000
    )
    predictions = f.predict(clf, knn, beta)
    args = acc.eval(predictions, 5)
    ans = ans + f"\nfusion\n{xc_metrics.format(*args)}"

    if save:
        fname = os.path.join(root, f"score.npz")
        save_npz(fname, retain_topk(predictions, k=top_k),
            compressed=False)
    return ans


if __name__ == '__main__':
    pass
