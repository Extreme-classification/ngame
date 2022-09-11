import numpy as np
import pickle
import os
import numba as nb
from xclib.utils.shortlist import Shortlist
from xclib.utils.clustering import cluster_balance, b_kmeans_dense


@nb.njit(parallel=True)
def map_dense(ind, mapping):
    out = np.full_like(ind, fill_value=0)
    nr, nc = ind.shape
    for i in nb.prange(nr):
        for j in range(nc):
            out[i, j] = mapping[ind[i, j]]
    return out


class ShortlistMIPS(Shortlist):
    """Get nearest labels using their embeddings
    * brute or HNSW algorithm for search
    * option to process label representations with label correlation matrix

    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int
        number of neighbors (same as efS)
        * may be useful if the NN search retrieve less number of labels
        * typically doesn't happen with HNSW etc.
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: int, optional, default=300
        construction parameter (Usually 300)
    efS: int, optional, default=300
        search parameter (Usually 300)
    num_threads: int, optional, default=18
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    """
    def __init__(self, method='hnswlib', num_neighbours=300, M=100, efC=300,
                 efS=300, space='cosine', verbose=True, num_threads=16):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads, space)
        self.valid_indices = None

    def fit(self, X, *args, **kwargs):
        ind = np.where(np.square(X).sum(axis=1) > 0)[0]
        self.valid_indices = ind
        X = X[self.valid_indices]
        super().fit(X)

    def query(self, X, *args, **kwargs):
        ind, sim = super().query(X)
        if self.valid_indices is not None:
            ind = map_dense(ind, self.valid_indices)
        return ind, sim

    def save(self, fname):
        metadata = {
            'valid_indices': self.valid_indices,
        }
        pickle.dump(metadata, open(fname+".metadata", 'wb'))
        super().save(fname+".index")

    def load(self, fname):
        self.index.load(fname+".index")
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self.valid_indices = obj['valid_indices']

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")


class ClusteringIndex(object):
    def __init__(self, num_instances, num_clusters, num_threads):
        self.num_instances = num_instances
        self.num_clusters = num_clusters
        self.num_threads = num_threads
        self.index = None
        self.avg_size = 1
        self.random_clustering()

    def random_clustering(self):
        self.index = []
        for i in range(self.num_instances):
           self.index.append([i])

    def update(self, X, num_clusters=None):
        assert self.num_instances == len(X)
        _nc = self.num_clusters if num_clusters is None else num_clusters
        self.index, _ = cluster_balance(
            X=X.copy(), 
            clusters=[np.arange(len(X), dtype='int')],
            num_clusters=_nc,
            splitter=b_kmeans_dense,
            num_threads=self.num_threads,
            verbose=True)
        self.avg_size = np.mean(list(map(len, self.index)))

    def query(self, idx):
        return self.index[idx]

    def save(self):
        pass

    def load(self):
        pass
