import logging
import os
import time
from scipy.sparse import issparse
import sys
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import xclib.evaluation.xc_metrics as xc_metrics
import sys
from .dataset import construct_dataset
from .collate_fn import collate
from .tracking import Tracking
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from xclib.utils.matrix import SMatrix
from .utils import filter_predictions


class ModelBase(object):
    """
    Base class for Deep extreme multi-label learning
    Arguments
    ---------
    params: NameSpace
        object containing parameters like learning rate etc.
    net: models.network.DeepXMLBase
        * DeepXMLs: network with a label shortlist
        * DeepXMLf: network with fully-connected classifier
    criterion: libs.loss._Loss
        to compute loss given y and y_hat
    optimizer: libs.optimizer.Optimizer
        to back-propagate and updating the parameters
    """

    def __init__(
        self,
        net,
        criterion,
        optimizer,
        schedular,
        model_dir,
        result_dir,
        *args,
        **kwargs
    ):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
        self.current_epoch = 0
        self.last_saved_epoch = -1
        self.model_dir = model_dir
        self.result_dir = result_dir
        self.last_epoch = 0
        self.model_fname = "model"
        self.logger = self.get_logger(name=self.model_fname)
        self.devices = -1 #self._create_devices(params.devices)
        self.tracking = Tracking()
        self.scaler = None

    def setup_amp(self, use_amp):
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def setup_devices(self):
        pass

    def _create_dataset(
        self,
        data_dir,
        fname,
        data=None,
        model_dir='',
        mode='train',
        sampling_params=None,
        normalize_features=True,
        normalize_labels=True,
        keep_invalid=False,
        feature_type='sparse',
        feature_indices=None,
        label_indices=None,
        label_feature_indices=None,
        batch_type='label',
        negative_sampler=None,
        max_len=-1,
        classifier_type='xc',
        precomputed_negatives=None,
        surrogate_mapping=None,
        **kwargs):
        """
        Create dataset as per given data and parameters
        Arguments
        ---------
        data_dir: str or None, optional, default=None
            load data from this directory when data is None
        fname_features: str
            load features from this file when data is None
        fname_labels: str or None, optional, default=None
            load labels from this file when data is None
        data: dict or None, optional, default=None
            directly use this this data when available
            * X: feature; Y: label (can be empty)
        mode: str, optional, default='predict'
            train or predict
        normalize_features: bool, optional, default=True
            Normalize data points to unit norm
        normalize_lables: bool, optional, default=False
            Normalize labels to convert in probabilities
            Useful in-case on non-binary labels
        feature_type: str, optional, default='sparse'
            sparse or dense features
        keep_invalid: bool, optional, default=False
            Don't touch data points or labels
        feature_indices: str or None, optional, default=None
            Train with selected features only (read from file)
        label_indices: str or None, optional, default=None
            Train for selected labels only (read from file)
        size_shortlist: int, optional, default=-1
            Size of shortlist (useful for datasets with a shortlist)
        shortlist_method: str, optional, default='static'
            static: fixed shortlist
            dynamic: dynamically generate shortlist
            hybrid: mixture of static and dynamic
        shorty: libs.shortlist.Shortlist or None, optional, default=None
            to generate a shortlist of labels
        surrogate_mapping: str, optional, default=None
            Re-map clusters as per given mapping
            e.g. when labels are clustered
        pretrained_shortlist: csr_matrix or None, default=None
            Shortlist for the dataset
        _type: str, optional, default='full'
            full: with full ground truth
            shortlist: with a shortlist
            tensor: with only features
        Returns
        -------
        dataset: Dataset
            return dataset created using given data and parameters
        """
        return construct_dataset(
            data_dir,
            fname,
            data=data,
            model_dir=model_dir,
            mode=mode,
            sampling_params=sampling_params,
            max_len=max_len,
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
            surrogate_mapping=surrogate_mapping,
            classifier_type=classifier_type
        )

    def create_batch_sampler(self, *args, **kwargs):
        return None

    def _create_data_loader(
        self,
        dataset,
        prefetch_factor=5,
        batch_size=128,
        feature_type='sparse',
        sampling_type='brute',
        classifier_type='xc',
        num_workers=4,
        shuffle=False, 
        **kwargs
    ):
        """
        Create data loader for given dataset
        Arguments
        ---------
        dataset: Dataset
            Dataset object
        batch_size: int, optional, default=128
            batch size
        num_workers: int, optional, default=4
            #workers in data loader
        shuffle: boolean, optional, default=False
            shuffle train data in each epoch
        mode: str, optional, default='predict'
            train or predict
        feature_type: str, optional, default='sparse'
            sparse or dense features
        classifier_type: str, optional, default='full'
            OVA or a classifier with shortlist
        """
        batch_sampler = self.create_batch_sampler(
                dataset, batch_size, shuffle
            )
        if batch_sampler is not None:
            dt_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                prefetch_factor=prefetch_factor,
                num_workers=num_workers,
                collate_fn=self._create_collate_fn(
                    feature_type, classifier_type, sampling_type))
        else:
            dt_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                prefetch_factor=prefetch_factor,
                num_workers=num_workers,
                shuffle=shuffle,
                collate_fn=self._create_collate_fn(
                    feature_type, classifier_type, sampling_type))
        return dt_loader

    def _create_collate_fn(self, feature_type, classifier_type, sampling_type):
        if classifier_type == 'siamese':
            op_feature_type = feature_type
        else:
            op_feature_type = None
        return collate(
                   feature_type, classifier_type, sampling_type, op_feature_type)

    def get_logger(self, name='SiameseXML', level=logging.INFO):
        """
        Return logging object!
        """
        logger = logging.getLogger(name)
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.propagate = False
        logging.Formatter(fmt='%(levelname)s:%(message)s')
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(level=level)
        return logger

    def _compute_loss(self, _pred, batch_data):
        """
            Compute loss for one classifier
        """
        _true = batch_data['Y'].to(_pred.get_device())
        return self.criterion(_pred, _true).to(self.devices[-1])

    def _step_amp(self, data_loader, precomputed_intermediate=False):
        """
        Training step (one pass over dataset)

        Arguments
        ---------
        data_loader: DataLoader
            data loader over train dataset
        batch_div: boolean, optional, default=False
            divide the loss with batch size?
            * useful when loss is sum over instances and labels
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features

        Returns
        -------
        loss: float
            mean loss over the train set
        """
        self.net.train()
        torch.set_grad_enabled(True)
        mean_loss = 0
        pbar = tqdm(data_loader)
        for batch_data in pbar:
            self.optimizer.zero_grad()
            batch_size = batch_data['batch_size']
            with torch.cuda.amp.autocast():
                out_ans = self.net.forward(batch_data, precomputed_intermediate)
                loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.schedular.step()
            pbar.set_description(
                f"loss: {loss.item():.5f}")
            del batch_data
        return mean_loss / data_loader.dataset.num_instances


    def _step(self, data_loader, precomputed_intermediate=False):
        """
        Training step (one pass over dataset)

        Arguments
        ---------
        data_loader: DataLoader
            data loader over train dataset
        batch_div: boolean, optional, default=False
            divide the loss with batch size?
            * useful when loss is sum over instances and labels
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features

        Returns
        -------
        loss: float
            mean loss over the train set
        """
        self.net.train()
        torch.set_grad_enabled(True)
        mean_loss = 0
        pbar = tqdm(data_loader)
        for batch_data in pbar:
            self.optimizer.zero_grad()
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data, precomputed_intermediate)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            pbar.set_description(
                f"loss: {loss.item():.5f}")
            del batch_data
        return mean_loss / data_loader.dataset.num_instances

    def _validate(self, data_loader, top_k=10):
        """
        predict for the given data loader
        * retruns loss and predicted labels

        Arguments
        ---------
        data_loader: DataLoader
            data loader over validation dataset
        top_k: int, optional, default=10
            Maintain top_k predictions per data point

        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        loss: float
            mean loss over the validation dataset
        """
        self.net.eval()
        top_k = min(top_k, data_loader.dataset.num_labels)
        torch.set_grad_enabled(False)
        mean_loss = 0
        predicted_labels = SMatrix(
            n_rows=data_loader.dataset.num_instances,
            n_cols=data_loader.dataset.num_labels,
            nnz=top_k)
        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            vals, ind = torch.topk(out_ans, k=top_k, dim=-1, sorted=False)
            predicted_labels.update_block(
                count, ind.cpu().numpy(), vals.cpu().numpy())
            count += batch_size
            del batch_data
        return predicted_labels.data(), \
            mean_loss / data_loader.dataset.num_instances

    def _fit(
        self,
        train_loader,
        validation_loader,
        init_epoch=0,
        num_epochs=10,
        validate_after=5
    ):
        """
        Train for the given data loader

        Arguments
        ---------
        train_loader: DataLoader
            data loader over train dataset
        validation_loader: DataLoader or None
            data loader over validation dataset
        model_dir: str
            save checkpoints etc. in this directory
        result_dir: str
            save logs etc in this directory
        init_epoch: int, optional, default=0
            start training from this epoch
            (useful when fine-tuning from a checkpoint)
        num_epochs: int
            #passes over the dataset
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features
        """
        for epoch in range(init_epoch, init_epoch+num_epochs):
            cond = self.dlr_step != -1 and epoch % self.dlr_step == 0
            if epoch != 0 and cond:
                self._adjust_parameters()
            batch_train_start_time = time.time()
            tr_avg_loss = self._step(train_loader)
            self.tracking.mean_train_loss.append(tr_avg_loss)
            batch_train_end_time = time.time()
            self.tracking.train_time = self.tracking.train_time + \
                batch_train_end_time - batch_train_start_time

            self.logger.info(
                "Epoch: {:d}, loss: {:.6f}, time: {:.2f} sec".format(
                    epoch, tr_avg_loss,
                    batch_train_end_time - batch_train_start_time))
            if validation_loader is not None and epoch % validate_after == 0:
                val_start_t = time.time()
                predicted_labels, val_avg_loss = self._validate(
                    validation_loader)
                val_end_t = time.time()
                self.tracking.validation_time = self.tracking.validation_time \
                    + val_end_t \
                    - val_start_t
                _prec, _ndcg = self.evaluate(
                    validation_loader.dataset.labels.Y, predicted_labels)
                self.tracking.mean_val_loss.append(val_avg_loss)
                self.tracking.val_precision.append(_prec)
                self.tracking.val_ndcg.append(_ndcg)
                self.logger.info("Model saved after epoch: {}".format(epoch))
                self.save_checkpoint(self.model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                self.logger.info(
                    "P@1: {:.2f}, loss: {:.6f}, time: {:.2f} sec".format(
                        _prec[0]*100, val_avg_loss, val_end_t-val_start_t))
            self.tracking.last_epoch += 1
        self.save_checkpoint(self.model_dir, epoch+1)
        self.tracking.save(
            os.path.join(self.result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {:.2f} sec, Validation time: {:.2f} sec"
            ", Shortlist time: {:.2f} sec, Model size: {:.2f} MB\n".format(
                self.tracking.train_time, self.tracking.validation_time,
                self.tracking.shortlist_time, self.model_size))

    def fit(
        self,
        data_dir,
        dataset,
        trn_fname,
        val_fname,
        trn_data=None,
        val_data=None,
        num_epochs=10,
        batch_size=128,
        num_workers=4,
        shuffle=False,
        init_epoch=0,
        keep_invalid=False,
        feature_indices=None,
        label_indices=None,
        normalize_features=True,
        normalize_labels=False,
        validate=False,
        validate_after=5,
        surrogate_mapping=None, **kwargs
    ):
        """
        Train for the given data
        * Also prints train time and model size

        Arguments
        ---------
        data_dir: str or None, optional, default=None
            load data from this directory when data is None
        model_dir: str
            save checkpoints etc. in this directory
        result_dir: str
            save logs etc in this directory
        dataset: str
            Name of the dataset
        learning_rate: float
            initial learning rate
        num_epochs: int
            #passes over the dataset
        data: dict or None, optional, default=None
            directly use this this data to train when available
            * X: feature; Y: label
        trn_feat_fname: str, optional, default='trn_X_Xf.txt'
            train features
        trn_label_fname: str, optional, default='trn_X_Y.txt'
            train labels
        val_feat_fname: str, optional, default='tst_X_Xf.txt'
            validation features (used only when validate is True)
        val_label_fname: str, optional, default='tst_X_Y.txt'
            validation labels (used only when validate is True)
        batch_size: int, optional, default=1024
            batch size in data loader
        num_workers: int, optional, default=6
            #workers in data loader
        shuffle: boolean, optional, default=True
            shuffle train data in each epoch
        init_epoch: int, optional, default=0
            start training from this epoch
            (useful when fine-tuning from a checkpoint)
        keep_invalid: bool, optional, default=False
            Don't touch data points or labels
        feature_indices: str or None, optional, default=None
            Train with selected features only (read from file)
        label_indices: str or None, optional, default=None
            Train for selected labels only (read from file)
        normalize_features: bool, optional, default=True
            Normalize data points to unit norm
        normalize_lables: bool, optional, default=False
            Normalize labels to convert in probabilities
            Useful in-case on non-binary labels
        validate: bool, optional, default=True
            validate using the given data if flag is True
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        feature_type: str, optional, default='sparse'
            sparse or dense features
        surrogate_mapping: str, optional, default=None
            Re-map clusters as per given mapping
            e.g. when labels are clustered
        """
        # Reset the logger to dump in train log file
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.result_dir, 'log_train.txt'))) 
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            trn_fname,
            data=trn_data,
            mode='train',
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices,
            surrogate_mapping=surrogate_mapping)
        train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle)
        # Compute and store representation if embeddings are fixed
        if self.freeze_intermediate:
            train_loader = self._create_data_loader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False)
            self.logger.info(
                "Computing and reusing coarse document embeddings"
                " to save computations.")
            data = {'X': None, 'Y': None}
            data['X'] = self.get_embeddings(
                data_dir=None,
                fname=None,
                data=train_dataset.features.data,
                return_coarse=True)
            data['Y'] = train_dataset.labels.data
            train_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                data=data,
                fname_features=None,
                feature_type='dense',
                mode='train',
                keep_invalid=True)  # Invalid labels already removed
            train_loader = self._create_data_loader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                classifier_type='full',
                shuffle=shuffle)
        self.logger.info("Loading validation data.")
        validation_loader = None
        if validate:
            validation_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                val_fname,
                data={'X': None, 'Y': None},
                mode='predict',
                keep_invalid=keep_invalid,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_indices=feature_indices,
                label_indices=label_indices,
                surrogate_mapping=surrogate_mapping)
            validation_loader = self._create_data_loader(
                validation_dataset,
                batch_size=batch_size,
                num_workers=num_workers)
        self._fit(train_loader, validation_loader, self.model_dir,
                  self.result_dir, init_epoch, num_epochs, validate_after)

    def _format_acc(self, acc):
        """
        Format accuracies (precision, ndcg) as string
        Useful in case of multiple
        """
        _res = ""
        if isinstance(acc, dict):
            for key, val in acc.items():
                _val = ','.join(map(lambda x: '%0.2f' % (x*100), val[0]))
                _res += "({}): {} ".format(key, _val)
        else:
            _val = ','.join(map(lambda x: '%0.2f' % (x*100), acc[0]))
            _res = "(clf): {}".format(_val)
        return _res

    def predict(
        self,
        data_dir,
        dataset,
        fname,
        data=None,
        batch_size=256,
        num_workers=6,
        keep_invalid=False,
        feature_indices=None,
        label_indices=None,
        top_k=50,
        normalize_features=True,
        normalize_labels=False,
        surrogate_mapping=None,
        feature_type='sparse',
        classifier_type='full', **kwargs
    ):
        """
        Predict for the given data
        * Also prints prediction time, precision and ndcg

        Arguments
        ---------
        data_dir: str or None, optional, default=None
            load data from this directory when data is None
        dataset: str
            Name of the dataset
        data: dict or None, optional, default=None
            directly use this this data when available
            * X: feature; Y: label (can be empty)
        tst_feat_fname: str, optional, default='tst_X_Xf.txt'
            load features from this file when data is None
        tst_label_fname: str, optional, default='tst_X_Y.txt'
            load labels from this file when data is None
            * can be dummy
        batch_size: int, optional, default=1024
            batch size in data loader
        num_workers: int, optional, default=6
            #workers in data loader
        keep_invalid: bool, optional, default=False
            Don't touch data points or labels
        feature_indices: str or None, optional, default=None
            Train with selected features only (read from file)
        label_indices: str or None, optional, default=None
            Train for selected labels only (read from file)
        top_k: int
            Maintain top_k predictions per data point
        normalize_features: bool, optional, default=True
            Normalize data points to unit norm
        normalize_lables: bool, optional, default=False
            Normalize labels to convert in probabilities
            Useful in-case on non-binary labels
        surrogate_mapping: str, optional, default=None
            Re-map clusters as per given mapping
            e.g. when labels are clustered
        feature_type: str, optional, default='sparse'
            sparse or dense features
        classifier_type: str, optional, default='full'
            OVA or a classifier with shortlist

        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        """
        self.logger.addHandler(
            logging.FileHandler(os.path.join(self.result_dir, 'log_predict.txt')))
        dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname=fname,
            data=data,
            mode='predict',
            feature_type=feature_type,
            size_shortlist=self.shortlist_size,
            _type=classifier_type,
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices,
            surrogate_mapping=surrogate_mapping)
        data_loader = self._create_data_loader(
            feature_type=feature_type,
            classifier_type=classifier_type,
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers)
        time_begin = time.time()
        predicted_labels = self._predict(data_loader, top_k, **kwargs)
        time_end = time.time()
        prediction_time = time_end - time_begin
        acc = self.evaluate(dataset.labels.data, predicted_labels)
        _res = self._format_acc(acc)
        self.logger.info(
            "Prediction time (total): {:.2f} sec.,"
            "Prediction time (per sample): {:.2f} msec., P@k(%): {:s}".format(
                prediction_time,
                prediction_time*1000/data_loader.dataset.num_instances, _res))
        return predicted_labels

    def _predict(self, data_loader, top_k, **kwargs):
        """
        Predict for the given data_loader

        Arguments
        ---------
        data_loader: DataLoader
            DataLoader object to create batches and iterate over it
        top_k: int
            Maintain top_k predictions per data point

        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        """
        self.net.eval()
        torch.set_grad_enabled(False)
        predicted_labels = SMatrix(
            n_rows=data_loader.dataset.num_instances,
            n_cols=data_loader.dataset.num_labels,
            nnz=top_k)
        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            vals, ind = torch.topk(out_ans, k=top_k, dim=-1, sorted=False)
            predicted_labels.update_block(
                count, ind.cpu().numpy(), vals.cpu().numpy())
            count += batch_size
        return predicted_labels.data()

    def _embeddings(
        self,
        data_loader,
        encoder=None,
        use_intermediate=False,
        fname_out=None,
        _dtype='float32'
    ):
        """
        Encode given data points
        * support for objects or files on disk

        Arguments
        ---------
        data_loader: DataLoader
            DataLoader object to create batches and iterate over it
        encoder: callable or None, optional, default=None
            use this function to encode given dataset
            * net.encode is used when None
        use_intermediate: boolean, optional, default=False
            return intermediate representation if True
        fname_out: str or None, optional, default=None
            load data from this file when data is None
        _dtype: str, optional, default='float32'
            data type of the encoded data
        """
        if encoder is None:
            self.logger.info("Using the default encoder.")
            encoder = self.net.encode
        self.net.eval()
        torch.set_grad_enabled(False)
        if fname_out is not None:  # Save to disk
            embeddings = np.memmap(
                fname_out, dtype=_dtype, mode='w+',
                shape=(data_loader.dataset.num_instances,
                       self.net.representation_dims))
        else:  # Keep in memory
            embeddings = np.zeros((
                data_loader.dataset.num_instances,
                self.net.representation_dims),
                dtype=_dtype)
        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = encoder(
                batch_data['X'], use_intermediate)
            embeddings[count:count+batch_size,
                       :] = out_ans.detach().cpu().numpy()
            count += batch_size
        torch.cuda.empty_cache()
        if fname_out is not None:  # Flush all changes to disk
            embeddings.flush()
        return embeddings

    def get_embeddings(
        self,
        encoder=None,
        data_dir=None,
        fname=None,
        data=None,
        batch_size=1024,
        num_workers=6,
        normalize=False,
        indices=None,
        fname_out=None,
        use_intermediate=False,
        feature_type='sparse', 
        **kwargs
    ):
        """
        Encode given data points
        * support for objects or files on disk

        Arguments
        ---------
        encoder: callable or None, optional, default=None
            use this function to encode given dataset
            * net.encode is used when None
        data_dir: str or None, optional, default=None
            load data from this directory when data is None
        fname: str or None, optional, default=None
            load data from this file when data is None
        data: csr_matrix or ndarray or None, optional, default=None
            directly use this this data when available
        batch_size: int, optional, default=1024
            batch size in data loader
        num_workers: int, optional, default=6
            #workers in data loader
        normalize: boolean, optioanl, default=False
            Normalize instances to unit l2-norm if True
        indices: list or None, optional or None
            Use only these feature indices; use all when None
        fname_out: str or None, optioanl, default=None
            save as memmap if filename is given
        use_intermediate: boolean, optional, default=False
            return intermediate representation if True
        feature_type: str, optional, default='sparse'
            feature type such as sparse/dense
        """
        if data is None:
            assert data_dir is not None and fname is not None, \
                "valid file path is required when data is not passed"
        dataset = self._create_dataset(
            data_dir,
            fname=fname,
            data=data,
            normalize_features=normalize,
            feature_type=feature_type,
            feature_indices=indices,
            classifier_type=None,
            **kwargs)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._create_collate_fn(
                feature_type=feature_type,
                classifier_type=None,
                sampling_type=None),
            shuffle=False)
        return self._embeddings(data_loader, encoder, use_intermediate, fname_out)

    def save_checkpoint(self, model_dir, epoch, do_purge=True):
        """
        Save checkpoint on disk
        * save network, optimizer and loss
        * filename: checkpoint_net_epoch.pkl for network

        Arguments:
        ---------
        model_dir: str
            save checkpoint into this directory
        epoch: int
            checkpoint after this epoch (used in file name)
        do_purge: boolean, optional, default=True
            delete old checkpoints beyond a point
        """
        checkpoint = {
            'epoch': epoch,
            'criterion': self.criterion.state_dict(),
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Useful if there are multiple parts of a model
        fname = {'net': 'checkpoint_net_{}.pkl'.format(epoch)}
        torch.save(checkpoint, os.path.join(model_dir, fname['net']))
        self.tracking.saved_checkpoints.append(fname)
        if do_purge:
            self.purge(model_dir)

    def load_checkpoint(self, model_dir, fname, epoch):
        """
        Load checkpoint from disk
        * load network, optimizer and loss
        * filename: checkpoint_net_epoch.pkl for network

        Arguments:
        ---------
        model_dir: str
            load checkpoint into this directory
        epoch: int
            checkpoint after this epoch (used in file name)
        """
        fname = os.path.join(model_dir, 'checkpoint_net_{}.pkl'.format(epoch))
        checkpoint = torch.load(open(fname, 'rb'))
        self.net.load_state_dict(checkpoint['net'])
        self.criterion.load_state_dict(checkpoint['criterion'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, model_dir, fname, *args):
        """
        Save model on disk
        * uses prefix: _network.pkl for network

        Arguments:
        ---------
        model_dir: str
            save model into this directory
        fname: str
            save model with this file name
        """
        fname = os.path.join(
            model_dir, fname+'_network.pkl')
        self.logger.info("Saving model at: {}".format(fname))
        state_dict = self.net.state_dict()
        torch.save(state_dict, fname)

    def load(self, model_dir, fname, *args):
        """
        Load model from disk
        * uses prefix: _network.pkl for network

        Arguments:
        ---------
        model_dir: str
            load model from this directory
        fname: str
            load model with this file name
        """
        fname_net = fname+'_network.pkl'
        state_dict = torch.load(
            os.path.join(model_dir, model_dir, fname_net))
        self.net.load_state_dict(state_dict)

    def purge(self, model_dir):
        """
        Remove checkpoints from disk
        * uses checkpoint_history to decide which checkpoint to delete
        * delete if #saved_checkpoints is more than a threshold; otherwise skip
        """
        if len(self.tracking.saved_checkpoints) \
                > self.tracking.checkpoint_history:
            fname = self.tracking.saved_checkpoints.pop(0)
            self.logger.info(
                "Purging network checkpoint: {}".format(fname['net']))
            os.remove(os.path.join(model_dir, fname['net']))

    def _evaluate(self, true_labels, predicted_labels):
        acc = xc_metrics.Metrics(true_labels)
        acc = acc.eval(predicted_labels.tocsr(), 5)
        return acc

    def evaluate(self, true_labels, predicted_labels, filter_map=None):
        """
        Compute precision and ndcg for given prediction matrix

        Arguments
        ---------
        true_labels: csr_matrix
            ground truth matrix
        predicted_labels: csr_matrix or dict
            predictions matrix (expect dictionary in case of multiple)
        filter_labels: np.ndarray or None, optional (default=None)
            filter predictions based on a given mapping
            
        Returns
        --------
        acc: list or dict of list
            return precision and ndcg
            * output dictionary uses same keys as input
        """
        if issparse(predicted_labels):
            return self._evaluate(
                true_labels, filter_predictions(predicted_labels, filter_map))
        else:  # Multiple set of predictions
            acc = {}
            for key, val in predicted_labels.items():
                acc[key] = self._evaluate(
                    true_labels, filter_predictions(val, filter_map))
            return acc

    @property
    def model_size(self):
        """
        Return model size (in MB)
        """
        return self.net.model_size
    
