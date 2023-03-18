import numpy as np
import pickle
import os
import numba as nb
from xclib.utils.shortlist import Shortlist
from xclib.utils.clustering import cluster_balance, b_kmeans_dense


def construct_shortlister(args):
    """Construct shortlister
    * used during predictions

    Arguments:
    ----------
    args: NameSpace
        parameters of the model with following inference methods
        * mips
          predict using a single nearest neighbor structure learned
          over label classifiers
        * dual_mips
          predict using two nearest neighbor structures learned
          over label embeddings and label classifiers
    """
    if args.inference_method == 'mips':  # Negative Sampling
        shortlister = ShortlistMIPS(
            method=args.ann_method,
            num_neighbours=args.num_nbrs,
            M=args.M,
            efC=args.efC,
            efS=args.efS,
            num_threads=args.ann_threads)
    elif args.inference_method == 'dual_mips':
        shortlister = DualShortlistMIPS(
            method=args.ann_method,
            num_neighbours=args.num_nbrs,
            M=args.M,
            efC=args.efC,
            efS=args.efS,
            num_threads=args.ann_threads)
    else:
        shortlister = None
    return shortlister


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
        try:
            metadata = {
                'valid_indices': self.valid_indices,
            }
            super().save(fname+".index")
        except ValueError or AttributeError:
            pass
        finally:            
            pickle.dump(metadata, open(fname+".metadata", 'wb'))

    def load(self, fname):
        self.index.load(fname+".index")
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self.valid_indices = obj['valid_indices']

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")
        if os.path.isfile(fname+".metadata"):
            os.remove(fname+".metadata")


class DualShortlistMIPS(object):
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
        self.mips_z = ShortlistMIPS(
            method, num_neighbours, M, efC, efS, space, verbose, num_threads)
        self.mips_w = ShortlistMIPS(
            method, num_neighbours, M, efC, efS, space, verbose, num_threads)

    def fit(self, Z, W):
        self.mips_z.fit(Z)
        self.mips_w.fit(W)

    def query(self, X, *args, **kwargs):
        return self.mips_z.query(X), self.mips_w.query(X)

    def save(self, fname):
        self.mips_z.save(fname+".z")
        self.mips_w.save(fname+".w")

    def load(self, fname):
        self.mips_z.load(fname+".z")
        self.mips_w.load(fname+".w")

    def purge(self, fname):
        self.mips_z.purge(fname+".z")
        self.mips_w.purge(fname+".w")

    @property
    def model_size(self):
        return self.mips_z.model_size + self.mips_w.model_size


class ClusteringIndex(object):
    def __init__(self, num_instances, num_clusters, num_threads, curr_steps):
        self.num_instances = num_instances
        self.num_clusters = num_clusters
        self.num_threads = num_threads
        self.index = None
        self.curr_steps = curr_steps
        self.step = 0
        self.avg_size = 1
        self.random_clustering()

    def update_state(self):
        if sum([i-1==self.step for i in self.curr_steps]) > 0:
            print(f"Doubling cluster size at: {self.step} to {2*self.num_instances/self.num_clusters}")
            # larger clusters => harder negatives
            self.num_clusters /= 2
        self.step += 1

    def random_clustering(self):
        self.index = []
        for i in range(self.num_instances):
           self.index.append([i])

    def update(self, X, num_clusters=None):
        assert self.num_instances == len(X)
        _nc = self.num_clusters if num_clusters is None else num_clusters
        self.index, _ = cluster_balance(
            X=X.astype('float32'), 
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
