import libs.xdataset as xdataset
import libs.sdataset as sdataset
from .dataset_base import DatasetTensor, DatasetBase


def _construct_dataset_class(classifier_t, batch_t, sampling_t):
    """
    Return the dataset class

    Arguments:
    ----------
    classifier_t: str or None
        - 'siamese': label-embeddings are treated as classifiers
        - 'xc': explcit classifiers
        - None: no classifiers; when only embeddings need to be computed
    batch_t: str 
        - 'doc': iterate over documents and sample negatives
        - 'lbl': iterate over labels and sample negatives
    sampling_t: str
        - implicit: (no-explicit negatives) in-batch sampling
        - explicit: explicitly sample negatives
        - brute: use all negatives
    """
    # assumes sampling is true
    if classifier_t == "siamese":
        if sampling_t == 'implicit':
            if batch_t == "doc":
                return sdataset.DatasetBDIS
            else: 
                return sdataset.DatasetBLIS
        elif sampling_t == 'explicit':
            raise NotImplementedError("")
        else:
            return DatasetBase
    elif classifier_t == "xc":
        if sampling_t == 'implicit':
            if batch_t == "doc":
                return xdataset.DatasetBDIS
            else:
                return xdataset.DatasetBLIS
        elif sampling_t == 'explicit':
            raise NotImplementedError("")
        else:
            return DatasetBase
    else:
        return DatasetTensor


def construct_dataset(data_dir,
                      fname=None,
                      data=None,
                      model_dir='',
                      mode='train',
                      sampling_params=None,
                      normalize_features=True,
                      normalize_labels=True,
                      keep_invalid=False,
                      feature_type='sparse',
                      classifier_type='xc',
                      feature_indices=None,
                      label_indices=None,
                      label_feature_indices=None,
                      batch_type='doc',
                      negative_sampler=None,
                      max_len=-1,
                      precomputed_negatives=None,
                      surrogate_mapping=None,
                      **kwargs):    
    try:
        sampling_type = sampling_params.type
        if sampling_type == 'brute':
            label_type = 'dense'
        else:
            label_type = 'sparse'
    except AttributeError:
        label_type = 'dense'
        sampling_type = None    

    if fname is None:
        fname = {'f_features': None,
                 'f_labels': None,
                 'f_label_features': None}


    cls = _construct_dataset_class(classifier_type, batch_type, sampling_type)
    return cls(data_dir=data_dir,
               **fname,
               data=data,
               model_dir=model_dir,
               mode=mode,
               label_type=label_type,
               max_len=max_len,
               sampling_params=sampling_params,
               normalize_features=normalize_features,
               normalize_labels=normalize_labels,
               keep_invalid=keep_invalid,
               feature_type=feature_type,
               feature_indices=feature_indices,
               label_indices=label_indices,
               label_feature_indices=label_feature_indices,
               negative_sampler=negative_sampler,
               batch_type=batch_type,
               precomputed_negatives=precomputed_negatives,
               surrogate_mapping=surrogate_mapping        
            )
