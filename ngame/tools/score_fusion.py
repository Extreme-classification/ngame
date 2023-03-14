import numpy as np
from sklearn.tree import DecisionTreeClassifier


class ScoreFusion():
    """
    A decision tree-based score fusion function to combine
    the classfier (clf) and shortlist based scores (knn)

    Arguments:
    ----------
    inv_psp: np.ndarray
        array containing inverse propensity values of labels    
    max_depth: int, optional (default=7)
        max depth of the decision trees    
    """
    def __init__(self, inv_psp, max_depth=7) -> None:
        self.inv_psp = inv_psp
        self.clf = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, val_pred_clf, val_pred_knn, val_lbl, ns=-1):
        if ns > 0 and ns < val_pred_clf.shape[0]:
            rand_ind = np.random.permutation(
                val_pred_clf.shape[0])[:ns]
            val_pred_knn = val_pred_knn[rand_ind]
            val_pred_clf = val_pred_clf[rand_ind]
            val_lbl = val_lbl[rand_ind]
        row, col = val_pred_clf.nonzero()
        scores = []
        for item in [val_pred_clf, val_pred_knn]:
            scores.append(np.array(item[row, col]).reshape(-1, 1))
        scores.append(self.inv_psp[col].reshape(-1, 1))
        scores = np.hstack(scores)
        targets = np.array(val_lbl[row, col]).ravel()
        self.clf.fit(scores, targets)

    def predict(self, pred_clf, pred_knn, beta=1):
        row, col = pred_clf.nonzero()
        scores = []
        for item in [pred_clf, pred_knn]:
            scores.append(np.array(item[row, col]).reshape(-1, 1))
        scores.append(self.inv_psp[col].reshape(-1, 1))
        scores = np.hstack(scores)        
        res = pred_clf.copy()
        res[row, col] = self.clf.predict_proba(scores)[:, 1]
        return beta*(res.tocsr() + pred_knn) + pred_clf
