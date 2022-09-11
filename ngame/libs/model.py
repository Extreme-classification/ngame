from functools import partial
from .model_base import ModelBase
from .dataset import DatasetBDIS, DatasetBLIS
from .dataset_base import DatasetTensor
import time
import os
import numpy as np
import torch
from scipy.sparse import issparse
from .collate_fn import collate_fn_seq_siamese
from torch.utils.data import DataLoader
from xclib.utils.sparse import csr_from_arrays
from .shortlist import ShortlistMIPS
from typing import Sized


class MySampler(torch.utils.data.Sampler[int]):
    def __init__(self, order):
        self.order = order.copy()

    def update_order(self, x):
        self.order[:] = x[:]

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


class ModelSiamese(ModelBase):
    """
    Models class for Siamese style models
    
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
    shorty: libs.shortlist.Shortlist
        to generate a shortlist of labels (random or from ANN)
    """

    def __init__(
        self,
        net,
        criterion,
        optimizer,
        schedular,
        model_dir,
        result_dir,
        freeze_intermediate=False,
        feature_type='sparse',
        shorty=None
    ):
        super().__init__(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            schedular=schedular,
            model_dir=model_dir,
            result_dir=result_dir,
            freeze_intermediate=freeze_intermediate,
            feature_type=feature_type,
        )
        self.shorty = shorty

    def _create_dataset(
        self,
        data_dir,
        fname,
        data=None,
        batch_type=None,
        _type=None,
        max_len=32,
        *args, **kwargs):
        """
        Create dataset as per given parameters
        """
        if batch_type == 'lbl':
            return DatasetBLIS(data_dir, **fname, max_len=max_len)
        elif batch_type == 'doc':
            return DatasetBDIS(data_dir, **fname, max_len=max_len)
        else:
            return DatasetTensor(data_dir, fname, data, _type='sequential')

    def _compute_loss_one(self, _pred, _true, _mask):
        # Compute loss for one classifier
        _true = _true.to(_pred.get_device())
        if _mask is not None:
            _mask = _mask.to(_true.get_device())
        return self.criterion(_pred, _true, _mask)

    def _compute_loss(self, out_ans, batch_data):
        """
        Compute loss for given pair of ground truth and logits
        """
        return self._compute_loss_one(
            out_ans, batch_data['Y'], batch_data['Y_mask'])

    def update_order(self, data_loader):
        data_loader.batch_sampler.sampler.update_order(
            data_loader.dataset.indices_permutation())

    def _fit(
        self,
        train_loader,
        validation_loader,
        init_epoch,
        num_epochs,
        validate_after,
        beta,
        use_intermediate_for_shorty,
        filter_map,
        sampling_warmup=20,
        sampling_update=5
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
        beta: float
            weightage of classifier when combining with shortlist scores
        use_intermediate_for_shorty: boolean
            use intermediate representation for negative sampling/ ANN search
        #TODO: complete documentation and implement with pre-computed features
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features
        """
        for epoch in range(init_epoch, init_epoch+num_epochs):
            if epoch >= sampling_warmup and epoch % sampling_update == 0:
                sampling_time = time.time()
                _X = self.get_embeddings(
                    data=train_loader.dataset.features.data,
                    encoder=self.net.encode_document,
                    batch_size=train_loader.batch_sampler.batch_size//2,
                    feature_type=train_loader.dataset.feature_type
                    )
                train_loader.dataset.update_shortlist(_X, 16384)
                self.update_order(train_loader)
                sampling_time = time.time() - sampling_time
                self.tracking.shortlist_time += sampling_time
                self.logger.info(
                "Updated sampler in time: {:.2f} sec".format(sampling_time))


            batch_train_start_time = time.time()
            tr_avg_loss = self._step(train_loader, batch_div=False)
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
                    train_loader, validation_loader, beta)
                val_end_t = time.time()
                _acc = self.evaluate(
                    validation_loader.dataset.labels,
                    predicted_labels, filter_map)
                self.tracking.validation_time = self.tracking.validation_time \
                    + val_end_t - val_start_t
                self.tracking.mean_val_loss.append(val_avg_loss)
                self.tracking.val_precision.append(_acc['knn'][0])
                self.tracking.val_ndcg.append(_acc['knn'][1])
                _acc = self._format_acc(_acc['knn'])
                self.logger.info("Model saved after epoch: {}".format(epoch))
                #self.save_checkpoint(self.model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                self.logger.info(
                    "P@1 (knn): {:s}, loss: {:s},"
                    " time: {:.2f} sec".format(
                        _acc, val_avg_loss,
                        val_end_t-val_start_t))
            self.tracking.last_epoch += 1

        #self.save_checkpoint(self.model_dir, epoch+1)
        self.tracking.save(os.path.join(self.result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {:.2f} sec, Validation time: {:.2f} sec, "
            "Shortlist time: {:.2f} sec, Model size: {:.2f} MB".format(
                self.tracking.train_time,
                self.tracking.validation_time,
                self.tracking.shortlist_time,
                self.model_size))

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
        normalize_features=True,
        normalize_labels=False,
        validate=False,
        beta=0.2,
        use_intermediate_for_shorty=True,
        validate_after=20,
        batch_type='doc',
        sampling_type=None,
        feature_type='sparse',
        shortlist_method=None,
        *args, **kwargs
    ):
        self.logger.info("Loading training data.")
        train_dataset = self._create_dataset(
            data_dir=os.path.join(data_dir, dataset),
            fname=trn_fname,
            data=trn_data,
            mode='train',
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_type=feature_type,
            shortlist_method=shortlist_method,
            size_shortlist=1,
            _type='embedding',
            batch_type=batch_type,
            shorty=self.shorty
            )
        train_loader = self._create_weighted_data_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle)

        validation_loader = None
        filter_map = None
        if validate:
            self.logger.info("Loading validation data.")
            validation_dataset = self._create_dataset(
                os.path.join(data_dir, dataset),
                fname=val_fname,
                data=val_data,
                mode='predict',
                feature_type=feature_type,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                shortlist_method=shortlist_method,
                size_shortlist=1,
                _type='embedding',
                batch_type='doc',
                shorty=self.shorty
                )
            print("Validation dataset is: ", validation_dataset)
            validation_loader = self._create_data_loader(
                validation_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False)
            filter_map = os.path.join(
                data_dir, dataset, 'filter_labels_test.txt')
            filter_map = np.loadtxt(filter_map).astype(np.int)
            val_ind = []
            valid_mapping = dict(
                zip(train_dataset._valid_labels,
                    np.arange(train_dataset.num_labels)))
            for i in range(len(filter_map)):
                if filter_map[i, 1] in valid_mapping:
                    filter_map[i, 1] = valid_mapping[filter_map[i, 1]]
                    val_ind.append(i)
            filter_map = filter_map[val_ind]

        self._fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            init_epoch=init_epoch,
            num_epochs=num_epochs,
            validate_after=validate_after,
            beta=beta,
            use_intermediate_for_shorty=use_intermediate_for_shorty,
            filter_map=filter_map)            
        train_time = self.tracking.train_time + self.tracking.shortlist_time
        return train_time, self.model_size

    def _create_weighted_data_loader(
        self,
        dataset,
        batch_size=128,
        max_len=32,
        num_workers=4,
        shuffle=False):
        """
            Create data loader for given dataset
        """
        order = dataset.indices_permutation()
        dt_loader = DataLoader(
            dataset,
            batch_sampler=torch.utils.data.sampler.BatchSampler(MySampler(order), batch_size, False),
            num_workers=num_workers,
            collate_fn=partial(collate_fn_seq_siamese, max_len=max_len)
            )
        return dt_loader

    def _validate(self, train_data_loader, data_loader, beta=0.2, gamma=0.5):
        self.net.eval()
        torch.set_grad_enabled(False)
        num_labels = data_loader.dataset.num_labels
        self.logger.info("Getting val document embeddings")
        val_doc_embeddings = self.get_embeddings(
            data=data_loader.dataset.features.data,
            encoder=self.net.encode_document,
            batch_size=data_loader.batch_size,
            feature_type=data_loader.dataset.feature_type
            )
        self.logger.info("Getting label embeddings")
        lbl_embeddings = self.get_embeddings(
            data=train_data_loader.dataset.lbl_features.data,
            encoder=self.net.encode_label,
            batch_size=data_loader.batch_size,
            feature_type=train_data_loader.dataset.feature_type
            )
        _shorty = ShortlistMIPS()
        _shorty.fit(lbl_embeddings)
        predicted_labels = {}
        ind, val = _shorty.query(val_doc_embeddings)
        predicted_labels['knn'] = csr_from_arrays(
            ind, val, shape=(data_loader.dataset.num_instances,
            num_labels+1))
        return self._strip_padding_label(predicted_labels, num_labels), \
            'NaN'

    def _strip_padding_label(self, mat, num_labels):
        stripped_vals = {}
        for key, val in mat.items():
            stripped_vals[key] = val[:, :num_labels].tocsr()
            del val
        return stripped_vals

    def evaluate(self, true_labels, predicted_labels, filter_map=None):
        def _filter(pred, mapping):
            if mapping is not None and len(mapping) > 0:
                pred[mapping[:, 0], mapping[:, 1]] = 0
                pred.eliminate_zeros()
            return pred
        if issparse(predicted_labels):
            return self._evaluate(
                true_labels, _filter(predicted_labels, filter_map))
        else:  # Multiple set of predictions
            acc = {}
            for key, val in predicted_labels.items():
                acc[key] = self._evaluate(
                    true_labels, _filter(val, filter_map))
            return acc