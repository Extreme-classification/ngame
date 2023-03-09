from functools import partial
from .model_base import ModelBase
from .dataset import XDatasetBDIS, SDatasetBDIS
from .dataset_base import DatasetTensor
import time
import os
import numpy as np
import torch
from scipy.sparse import issparse
from .collate_fn import collate_fn_seq_siamese, collate_fn_seq_xc
from torch.utils.data import DataLoader
from xclib.utils.sparse import csr_from_arrays, normalize
from .shortlist import ShortlistMIPS
from tqdm import tqdm
from xclib.utils.matrix import SMatrix
import logging


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
        use_amp=False,
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
            use_amp=use_amp,
            feature_type=feature_type,
        )
        self.shorty = shorty
        self.memory_bank = None

    def _create_dataset(
        self,
        data_dir,
        fname,
        sampling_params=None,
        data=None,
        batch_type=None,
        _type=None,
        max_len=32,
        *args, **kwargs):
        """
        Create dataset as per given parameters
        """
        if batch_type == 'lbl':
            return SDatasetBLIS(
                data_dir, **fname, sampling_params=sampling_params, max_len=max_len)
        elif batch_type == 'doc':
            return SDatasetBDIS(
                data_dir, **fname, sampling_params=sampling_params, max_len=max_len)
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
                out_ans, rep = self.net.forward(
                    batch_data, precomputed_intermediate)
                loss = self._compute_loss(out_ans, batch_data)
            if self.memory_bank is not None:
                ind = batch_data['indices']
                self.memory_bank[ind] = rep.detach().cpu().numpy()
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
            out_ans, rep = self.net.forward(
                batch_data, precomputed_intermediate)
            if self.memory_bank is not None:
                ind = batch_data['indices']
                self.memory_bank[ind] = rep.detach().cpu().numpy()
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            pbar.set_description(
                f"loss: {loss.item():.5f}")
            del batch_data
        return mean_loss / data_loader.dataset.num_instances

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
        sampling_params
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
        smp_warmup = sampling_params.sampling_curr_epochs[0]
        smp_refresh_interval = sampling_params.sampling_refresh_interval

        for epoch in range(init_epoch, init_epoch+num_epochs):
            if epoch >= smp_warmup and epoch % smp_refresh_interval == 0:
                sampling_time = time.time()
                if self.memory_bank is None:
                    _X = self.get_embeddings(
                        data=train_loader.dataset.features.data,
                        encoder=self.net.encode_document,
                        batch_size=train_loader.batch_sampler.batch_size//2,
                        feature_type=train_loader.dataset.feature_type
                        )
                else:
                    _X = self.memory_bank
                train_loader.dataset.update_shortlist(_X)
                sampling_time = time.time() - sampling_time
                self.tracking.shortlist_time += sampling_time
                self.logger.info(
                "Updated sampler in time: {:.2f} sec".format(sampling_time))


            batch_train_start_time = time.time()
            if self.scaler is None:
                tr_avg_loss = self._step(train_loader)
            else:
                tr_avg_loss = self._step_amp(train_loader)
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
                self.save_checkpoint(self.model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                self.logger.info(
                    "P@1 (knn): {:s}, loss: {:s},"
                    " time: {:.2f} sec".format(
                        _acc, val_avg_loss,
                        val_end_t-val_start_t))
            self.tracking.last_epoch += 1
            self.update_order(train_loader)
            train_loader.dataset.update_state()
        self.save_checkpoint(self.model_dir, epoch+1)
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
        sampling_params,
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
        max_len=32,
        feature_type='sparse',
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
            sampling_params=sampling_params,
            _type='embedding',
            max_len=max_len,
            batch_type=batch_type,
            shorty=self.shorty
            )
        train_loader = self._create_weighted_data_loader(
            train_dataset,
            batch_size=batch_size,
            max_len=max_len,
            num_workers=num_workers,
            shuffle=shuffle)
        if sampling_params.sampling_async:
            self.memory_bank = np.zeros(
                (len(train_dataset), self.net.representation_dims),
                dtype='float32')
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
                size_shortlist=1,
                _type='embedding',
                max_len=max_len,
                batch_type='doc',
                shorty=self.shorty
                )
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
            sampling_params=sampling_params,
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
        prefetch_factor=5,
        shuffle=False):
        """
            Create data loader for given dataset
        """
        order = dataset.indices_permutation()
        dt_loader = DataLoader(
            dataset,
            batch_sampler=torch.utils.data.sampler.BatchSampler(MySampler(order), batch_size, False),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
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
        

class ModelSShortlist(ModelBase):
    """
    Models with label shortlist
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
        to generate a shortlist of labels (typically an ANN structure)
        * same shortlist method is used during training and prediction
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
        use_amp=False,
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
            use_amp=use_amp,
            feature_type=feature_type,
        )
        self.shorty = shorty
        self.memory_bank = None


    def _compute_loss_one(self, _pred, _true, _mask):
        """
        Compute loss for one classifier
        """
        _true = _true.to(_pred.get_device())
        if _mask is not None:
            _mask = _mask.to(_true.get_device())
        return self.criterion(_pred, _true, _mask).to(_true.get_device())

    def _compute_loss(self, out_ans, batch_data):
        """
        Compute loss for given pair of ground truth and logits
        """
        return self._compute_loss_one(
            out_ans, batch_data['Y'], batch_data['Y_mask'])

    def _combine_scores(self, logit, sim, beta):
        """
        Combine scores of label classifier and shortlist
        score = beta*sigmoid(logit) + (1-beta)*sigmoid(sim)
        """
        return beta*sigmoid(logit, copy=True) \
            + (1-beta)*sigmoid(sim, copy=True)

    def _strip_padding_label(self, mat, num_labels):
        stripped_vals = {}
        for key, val in mat.items():
            stripped_vals[key] = val[:, :num_labels].tocsr()
            del val
        return stripped_vals

    def _fit_shorty(self, features, labels, doc_embeddings=None,
                    lbl_embeddings=None, use_intermediate=True,
                    feature_type='sparse'):
        """
        Train the ANN Structure with given data
        * Support for pre-computed features
        * Features are computed when pre-computed features are not available
        Arguments
        ---------
        features: np.ndarray or csr_matrix or None
            features for given data (used when doc_embeddings is None)
        labels: csr_matrix
            ground truth matrix for given data
        doc_embeddings: np.ndarray or None, optional, default=None
            pre-computed features; features are computed when None
        lbl_embeddings: np.ndarray or None, optional, default=None
            pre-computed label features; features are computed when None
        use_intermediate: boolean, optional, default=True
            use intermediate representation if True
        feature_type: str, optional, default='sparse'
            sparse or dense features
        """
        if doc_embeddings is None:
            doc_embeddings = self.get_embeddings(
                data=features,
                feature_type=feature_type,
                use_intermediate=use_intermediate)
        if isinstance(self.shorty, ShortlistMIPS):
            self.shorty.fit(X=lbl_embeddings, Y=labels)
        else:
            self.shorty.fit(X=doc_embeddings, Y=labels, Yf=lbl_embeddings)

    def _update_shortlist(self, dataset, use_intermediate=True, mode='train',
                          flag=True):
        if flag:
            if isinstance(dataset.features, DenseFeatures) and use_intermediate:
                self.logger.info("Using pre-trained embeddings for shortlist.")
                doc_embeddings = dataset.features.data
                lbl_embeddings = dataset.label_features.data
            else:
                doc_embeddings = self.get_embeddings(
                    data=dataset.features.data,
                    encoder=self.net.encode_document,
                    use_intermediate=use_intermediate)
                lbl_embeddings = self.get_embeddings(
                    data=dataset.label_features.data,
                    encoder=self.net.encode_label,
                    use_intermediate=use_intermediate)
            if mode == 'train':
                self.shorty.reset()
                self._fit_shorty(
                    features=None,
                    labels=dataset.labels.data,
                    lbl_embeddings=lbl_embeddings,
                    doc_embeddings=doc_embeddings)
            dataset.update_shortlist(
                *self._predict_shorty(doc_embeddings))

    def _predict_shorty(self, doc_embeddings):
        """
        Get nearest neighbors (and sim) for given document embeddings
        Arguments
        ---------
        doc_embeddings: np.ndarray
            embeddings/encoding for the data points
        Returns
        -------
        neighbors: np.ndarray
            indices of nearest neighbors
        sim: np.ndarray
            similarity with nearest neighbors
        """
        return self.shorty.query(doc_embeddings)

    def _update_predicted_shortlist(self, count, batch_size, predicted_labels,
                                    batch_out, batch_data):
        _indices = batch_data['Y_s'].numpy()
        _knn_score = batch_data['Y_sim'].numpy()
        _clf_score = batch_out.data.cpu().numpy()
        predicted_labels['clf'].update_block(count, _indices, _clf_score)
        predicted_labels['knn'].update_block(count, _indices, _knn_score)

    def _validate(self, data_loader, beta=0.2, top_k=20):
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
        def _predict_ova(X, clf, k=20, batch_size=32, device="cuda", return_sparse=True):
            """Predictions in brute-force manner"""
            if not torch.cuda.is_available():
                device = torch.device("cpu")
            torch.set_grad_enabled(False)
            num_instances, num_labels = len(X), len(clf)
            batches = np.array_split(range(num_instances), num_instances//batch_size)
            output = SMatrix(
                n_rows=num_instances,
                n_cols=num_labels,
                nnz=k)
            X = torch.from_numpy(X)        
            clf = torch.from_numpy(clf).to(device).T   
            for ind in tqdm(batches):
                s_ind, e_ind = ind[0], ind[-1] + 1
                _X = X[s_ind: e_ind].to(device)
                ans = _X @ clf
                vals, ind = torch.topk(
                    ans, k=k, dim=-1, sorted=True)
                output.update_block(
                    s_ind, ind.cpu().numpy(), vals.cpu().numpy())
                del _X
            if return_sparse:
                return output.data()
            else:
                return output.data('dense')[0]

        self.net.eval()
        torch.set_grad_enabled(False)
        num_labels = data_loader.dataset.num_labels
        self.logger.info("Getting val document embeddings")
        val_doc_embeddings = self.get_embeddings(
            data=data_loader.dataset.features.data,
            encoder=self.net.encode_document,
            batch_size=1024, #data_loader.batch_size,
            feature_type=data_loader.dataset.feature_type
            )
        self.logger.info("Getting label embeddings")
        lbl_embeddings = self.net.get_clf_weights()
        predicted_labels = {}
        predicted_labels['knn'] = _predict_ova(val_doc_embeddings, lbl_embeddings)
        # _shorty = ShortlistMIPS()
        # _shorty.fit(lbl_embeddings)
        # predicted_labels = {}
        # ind, val = _shorty.query(val_doc_embeddings)
        # predicted_labels['knn'] = csr_from_arrays(
        #     ind, val, shape=(data_loader.dataset.num_instances,
        #     num_labels+1))
        return self._strip_padding_label(predicted_labels, num_labels), \
            'NaN'

        # num_labels = data_loader.dataset.num_labels
        # num_instances = data_loader.dataset.num_instances
        # mean_loss = 0
        # predicted_labels = {}
        # predicted_labels['knn'] = SMatrix(
        #     n_rows=num_instances,
        #     n_cols=num_labels,
        #     nnz=top_k)

        # predicted_labels['clf'] = SMatrix(
        #     n_rows=num_instances,
        #     n_cols=num_labels,
        #     nnz=top_k)

        # count = 0
        # for batch_data in tqdm(data_loader):
        #     batch_size = batch_data['batch_size']
        #     out_ans = self.net.forward(batch_data)
        #     loss = self._compute_loss(out_ans, batch_data)/batch_size
        #     mean_loss += loss.item()*batch_size
        #     self._update_predicted_shortlist(
        #         count, batch_size, predicted_labels, out_ans, batch_data)
        #     count += batch_size
        # for k, v in predicted_labels.items():
        #     predicted_labels[k] = v.data()
        # predicted_labels['ens'] = self._combine_scores(
        #     predicted_labels['clf'], predicted_labels['knn'], beta)
        # return predicted_labels, mean_loss / num_instances

    def update_order(self, data_loader):
        data_loader.batch_sampler.sampler.update_order(
            data_loader.dataset.indices_permutation())

    def _fit(self,
             train_loader,
             validation_loader,
             model_dir,
             result_dir,
             init_epoch,
             num_epochs,
             validate_after,
             beta,
             sampling_params,
             use_intermediate_for_shorty,
             precomputed_intermediate,
             filter_map):
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
        precomputed_intermediate: boolean, optional, default=False
            if precomputed intermediate features are already available
            * avoid recomputation of intermediate features
        filter_labels: #TODO
        """
        smp_warmup = sampling_params.sampling_curr_epochs[0]
        smp_refresh_interval = sampling_params.sampling_refresh_interval

        for epoch in range(init_epoch, init_epoch+num_epochs):
            if epoch >= smp_warmup and epoch % smp_refresh_interval == 0:
                sampling_time = time.time()
                if self.memory_bank is None:
                    _X = self.get_embeddings(
                        data=train_loader.dataset.features.data,
                        encoder=self.net.encode_document,
                        batch_size=train_loader.batch_sampler.batch_size//2,
                        feature_type=train_loader.dataset.feature_type
                        )
                else:
                    _X = self.memory_bank
                train_loader.dataset.update_shortlist(_X)
                sampling_time = time.time() - sampling_time
                self.tracking.shortlist_time += sampling_time
                self.logger.info(
                "Updated sampler in time: {:.2f} sec".format(sampling_time))


            batch_train_start_time = time.time()
            if self.scaler is None:
                tr_avg_loss = self._step(train_loader)
            else:
                tr_avg_loss = self._step_amp(train_loader)
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
                    validation_loader, beta)
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
                self.save_checkpoint(self.model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                self.logger.info(
                    "P@1 (knn): {:s}, loss: {:s},"
                    " time: {:.2f} sec".format(
                        _acc, val_avg_loss,
                        val_end_t-val_start_t))
            self.tracking.last_epoch += 1
            self.update_order(train_loader)
            train_loader.dataset.update_state()
        self.save_checkpoint(self.model_dir, epoch+1)
        self.tracking.save(os.path.join(self.result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {:.2f} sec, Validation time: {:.2f} sec, "
            "Shortlist time: {:.2f} sec, Model size: {:.2f} MB".format(
                self.tracking.train_time,
                self.tracking.validation_time,
                self.tracking.shortlist_time,
                self.model_size))




        # for epoch in range(init_epoch, init_epoch+num_epochs):
        #     cond = self.dlr_step != -1 and epoch % self.dlr_step == 0
        #     if epoch != 0 and cond:
        #         self._adjust_parameters()
        #     batch_train_start_time = time.time()
        #     if epoch % self.retrain_hnsw_after == 0:
        #         self.logger.info(
        #             "Updating shortlist at epoch: {}".format(epoch))
        #         shorty_start_t = time.time()
        #         self._update_shortlist(
        #             dataset=train_loader.dataset,
        #             use_intermediate=use_intermediate_for_shorty,
        #             mode='train',
        #             flag=self.shorty is not None)
        #         if validation_loader is not None:
        #             self._update_shortlist(
        #                 dataset=validation_loader.dataset,
        #                 use_intermediate=use_intermediate_for_shorty,
        #                 mode='predict',
        #                 flag=self.shorty is not None)
        #         shorty_end_t = time.time()
        #         self.logger.info("ANN train time: {0:.2f} sec".format(
        #             shorty_end_t - shorty_start_t))
        #         self.tracking.shortlist_time = self.tracking.shortlist_time \
        #             + shorty_end_t - shorty_start_t
        #         batch_train_start_time = time.time()
        #     tr_avg_loss = self._step(
        #         train_loader, batch_div=True,
        #         precomputed_intermediate=precomputed_intermediate)
        #     self.tracking.mean_train_loss.append(tr_avg_loss)
        #     batch_train_end_time = time.time()
        #     self.tracking.train_time = self.tracking.train_time + \
        #         batch_train_end_time - batch_train_start_time

        #     self.logger.info(
        #         "Epoch: {:d}, loss: {:.6f}, time: {:.2f} sec".format(
        #             epoch, tr_avg_loss,
        #             batch_train_end_time - batch_train_start_time))
        #     if validation_loader is not None and epoch % validate_after == 0:
        #         val_start_t = time.time()
        #         predicted_labels, val_avg_loss = self._validate(
        #             validation_loader, beta, self.shortlist_size)
        #         val_end_t = time.time()
        #         _acc = self.evaluate(
        #             validation_loader.dataset.labels.data,
        #             predicted_labels, filter_labels)
        #         self.tracking.validation_time = self.tracking.validation_time \
        #             + val_end_t - val_start_t
        #         self.tracking.mean_val_loss.append(val_avg_loss)
        #         self.tracking.val_precision.append(_acc['ens'][0])
        #         self.tracking.val_ndcg.append(_acc['ens'][1])
        #         self.logger.info("Model saved after epoch: {}".format(epoch))
        #         self.save_checkpoint(model_dir, epoch+1)
        #         self.tracking.last_saved_epoch = epoch
        #         _res = self._format_acc(_acc)
        #         self.logger.info(
        #             "P@k {:s}, loss: {:.6f}, time: {:.2f} sec".format(
        #                 _res, val_avg_loss, val_end_t-val_start_t))
        #     self.tracking.last_epoch += 1

        # self.save_checkpoint(model_dir, epoch+1)
        # self.tracking.save(os.path.join(result_dir, 'training_statistics.pkl'))
        # self.logger.info(
        #     "Training time: {:.2f} sec, Validation time: {:.2f} sec, "
        #     "Shortlist time: {:.2f} sec, Model size: {:.2f} MB".format(
        #         self.tracking.train_time,
        #         self.tracking.validation_time,
        #         self.tracking.shortlist_time,
        #         self.model_size))

    def _create_dataset(
        self,
        data_dir,
        fname,
        sampling_params=None,
        data=None,
        batch_type=None,
        _type=None,
        max_len=32,
        *args, **kwargs):
        """
        Create dataset as per given parameters
        """
        if batch_type == 'lbl':
            return DatasetBLIS(
                data_dir, **fname, sampling_params=sampling_params, max_len=max_len)
        elif batch_type == 'doc':
            return XDatasetBDIS(
                data_dir, **fname, sampling_params=sampling_params, max_len=max_len)
        else:
            return DatasetTensor(data_dir, fname, data, _type='sequential')
    
    def init_classifier(self, dataset, batch_size=128):
        lbl_embeddings = self.get_embeddings(
            data=dataset.lbl_features.data,
            encoder=self.net.encode_label,
            batch_size=batch_size,
            feature_type=dataset.feature_type
            )
        self.net.initialize_classifier(
            np.vstack(
            [normalize(lbl_embeddings),
             np.zeros((1, self.net.representation_dims))])
            )

    def _create_data_loader(
        self,
        dataset,
        batch_size=128,
        max_len=32,
        num_workers=4,
        prefetch_factor=5,
        shuffle=False):
        """
            Create data loader for given dataset
        """
        order = dataset.indices_permutation()
        dt_loader = DataLoader(
            dataset,
            batch_sampler=torch.utils.data.sampler.BatchSampler(MySampler(order), batch_size, False),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=partial(collate_fn_seq_xc, max_len=max_len)
            )
        return dt_loader

    # def _step_amp(self, data_loader, precomputed_intermediate=False):
    #     """
    #     Training step (one pass over dataset)

    #     Arguments
    #     ---------
    #     data_loader: DataLoader
    #         data loader over train dataset
    #     batch_div: boolean, optional, default=False
    #         divide the loss with batch size?
    #         * useful when loss is sum over instances and labels
    #     precomputed_intermediate: boolean, optional, default=False
    #         if precomputed intermediate features are already available
    #         * avoid recomputation of intermediate features

    #     Returns
    #     -------
    #     loss: float
    #         mean loss over the train set
    #     """
    #     self.net.train()
    #     torch.set_grad_enabled(True)
    #     mean_loss = 0
    #     pbar = tqdm(data_loader)
    #     for batch_data in pbar:
    #         self.optimizer.zero_grad()
    #         batch_size = batch_data['batch_size']
    #         with torch.cuda.amp.autocast():
    #             out_ans, rep = self.net.forward(
    #                 batch_data, precomputed_intermediate)
    #             loss = self._compute_loss(out_ans, batch_data)
    #         if self.memory_bank is not None:
    #             ind = batch_data['indices']
    #             self.memory_bank[ind] = rep.detach().cpu().numpy()
    #         mean_loss += loss.item()*batch_size
    #         self.scaler.scale(loss).backward()
    #         self.scaler.step(self.optimizer)
    #         self.scaler.update()
    #         self.schedular.step()
    #         pbar.set_description(
    #             f"loss: {loss.item():.5f}")
    #         del batch_data
    #     return mean_loss / data_loader.dataset.num_instances


    # def _step(self, data_loader, precomputed_intermediate=False):
    #     """
    #     Training step (one pass over dataset)

    #     Arguments
    #     ---------
    #     data_loader: DataLoader
    #         data loader over train dataset
    #     batch_div: boolean, optional, default=False
    #         divide the loss with batch size?
    #         * useful when loss is sum over instances and labels
    #     precomputed_intermediate: boolean, optional, default=False
    #         if precomputed intermediate features are already available
    #         * avoid recomputation of intermediate features

    #     Returns
    #     -------
    #     loss: float
    #         mean loss over the train set
    #     """
    #     self.net.train()
    #     torch.set_grad_enabled(True)
    #     mean_loss = 0
    #     pbar = tqdm(data_loader)
    #     for batch_data in pbar:
    #         self.optimizer.zero_grad()
    #         batch_size = batch_data['batch_size']
    #         out_ans, rep = self.net.forward(
    #             batch_data, precomputed_intermediate)
    #         if self.memory_bank is not None:
    #             ind = batch_data['indices']
    #             self.memory_bank[ind] = rep.detach().cpu().numpy()
    #         loss = self._compute_loss(out_ans, batch_data)
    #         mean_loss += loss.item()*batch_size
    #         loss.backward()
    #         self.optimizer.step()
    #         self.schedular.step()
    #         pbar.set_description(
    #             f"loss: {loss.item():.5f}")
    #         del batch_data
    #     return mean_loss / data_loader.dataset.num_instances

    def fit(self,
            data_dir,
            model_dir,
            result_dir,
            dataset,
            num_epochs,
            trn_fname,
            val_fname,
            sampling_params,
            trn_data=None,
            val_data=None,
            batch_type='doc',
            max_len=32,
            batch_size=128,
            num_workers=4,
            shuffle=False,
            init_epoch=0,
            normalize_features=True,
            normalize_labels=False,
            validate=False,
            beta=0.2,
            use_intermediate_for_shorty=True,
            feature_type='sparse',
            shortlist_method='static',
            validate_after=5,
            surrogate_mapping=None,
            trn_pretrained_shortlist=None,
            val_pretrained_shortlist=None, **kwargs):
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
        lbl_feat_fname: str, optional, default='lbl_X_Xf.txt'
            label features
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
        beta: float, optional, default=0.5
            weightage of classifier when combining with shortlist scores
        use_intermediate_for_shorty: boolean, optional, default=True
            use intermediate representation for negative sampling/ANN
        shortlist_method: str, optional, default='static'
            static: fixed shortlist
            dynamic: dynamically generate shortlist
            hybrid: mixture of static and dynamic
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        surrogate_mapping: str, optional, default=None
            Re-map clusters as per given mapping
            e.g. when labels are clustered
        feature_type: str, optional, default='sparse'
            sparse or dense features
        trn_pretrained_shortlist: csr_matrix or None, default=None
            Shortlist for train dataset
        val_pretrained_shortlist: csr_matrix or None, default=None
            Shortlist for validation dataset
        """

        pretrained_shortlist = trn_pretrained_shortlist is not None
        # Reset the logger to dump in train log file
        self.logger.addHandler(
            logging.FileHandler(os.path.join(result_dir, 'log_train.txt')))
        self.logger.info("Loading training data.")

        train_dataset = self._create_dataset(
            data_dir=os.path.join(data_dir, dataset),
            fname=trn_fname,
            data=trn_data,
            mode='train',
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_type=feature_type,
            sampling_params=sampling_params,
            _type='embedding',
            max_len=max_len,
            batch_type=batch_type,
            shorty=self.shorty
            )
        _train_dataset = train_dataset
        self.init_classifier(train_dataset)

        train_loader = self._create_data_loader(
            train_dataset,
            # feature_type=train_dataset.feature_type,
            # classifier_type='shortlist',
            batch_size=batch_size,
            max_len=max_len,
            prefetch_factor=5,
            num_workers=num_workers,
            shuffle=shuffle)
        precomputed_intermediate = False
        # if self.freeze_intermediate or not self.update_shortlist:
        #     self.retrain_hnsw_after = 10000

        # No need to update embeddings
        # if self.freeze_intermediate and feature_type != 'dense':
        #     precomputed_intermediate = True
        #     self.logger.info(
        #         "Computing and reusing intermediate document embeddings "
        #         "to save computations.")
        #     data = {'X': None, 'Y': None}
        #     data['X'] = self.get_embeddings(
        #         data_dir=None,
        #         encoder=self.net.encode_label,
        #         fname=None,
        #         data=train_dataset.features.data,
        #         use_intermediate=True)
        #     data['Yf'] = self.get_embeddings(
        #         data_dir=None,
        #         encoder=self.net.encode_document,
        #         fname=None,
        #         data=train_dataset.label_features.data,
        #         use_intermediate=True)
        #     data['Y'] = train_dataset.labels.data
        #     _train_dataset = train_dataset
        #     train_dataset = self._create_dataset(
        #         os.path.join(data_dir, dataset),
        #         data=data,
        #         fname_features=None,
        #         fname_label_features=None,
        #         mode='train',
        #         normalize_features=False,  # do not normalize dense features
        #         shortlist_method=shortlist_method,
        #         size_shortlist=self.shortlist_size,
        #         feature_type='dense',
        #         pretrained_shortlist=trn_pretrained_shortlist,
        #         keep_invalid=True,   # Invalid labels already removed
        #         _type='shortlist')
        #     train_loader = self._create_data_loader(
        #         train_dataset,
        #         feature_type='dense',
        #         classifier_type='shortlist',
        #         batch_size=batch_size,
        #         num_workers=num_workers,
        #         shuffle=shuffle)
        self.logger.info("Loading validation data.")
        validation_loader = None
        filter_map = None
        if validate:
            validation_dataset = self._create_dataset(
                data_dir=os.path.join(data_dir, dataset),
                fname=val_fname,
                data=val_data,
                mode='test',
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_type=feature_type,
                sampling_params=sampling_params,
                _type='embedding',
                max_len=max_len,
                batch_type=batch_type,
                shorty=self.shorty
                )
            validation_loader = self._create_data_loader(
                validation_dataset,
                # feature_type=train_dataset.feature_type,
                # classifier_type='shortlist',
                batch_size=batch_size,
                max_len=max_len,
                prefetch_factor=5,
                num_workers=num_workers,
                shuffle=False)
            filter_map = os.path.join(
                data_dir, dataset, 'filter_labels_test.txt')
            filter_map = np.loadtxt(filter_map).astype(np.int)
            val_ind = []
            valid_mapping = dict(
                zip(_train_dataset._valid_labels,
                    np.arange(train_dataset.num_labels)))
            for i in range(len(filter_map)):
                if filter_map[i, 1] in valid_mapping:
                    filter_map[i, 1] = valid_mapping[filter_map[i, 1]]
                    val_ind.append(i)
            filter_map = filter_map[val_ind]
        del _train_dataset
        self._fit(train_loader, validation_loader, model_dir,
                  result_dir, init_epoch, num_epochs,
                  validate_after, beta, sampling_params,
                  use_intermediate_for_shorty,
                  precomputed_intermediate, filter_map)
        train_time = self.tracking.train_time + self.tracking.shortlist_time
        return train_time, self.model_size

    def _predict(self, data_loader, top_k, use_intermediate_for_shorty, beta):
        """
        Predict for the given data_loader
        Arguments
        ---------
        data_loader: DataLoader
            DataLoader object to create batches and iterate over it
        top_k: int
            Maintain top_k predictions per data point
        use_intermediate_for_shorty: bool
            use intermediate representation for negative sampling/ANN
        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        """
        self.logger.info("Loading test data.")
        self.net.eval()
        num_labels = data_loader.dataset.num_labels

        torch.set_grad_enabled(False)
        self.logger.info("Fetching shortlist.")
        self._update_shortlist(
            dataset=data_loader.dataset,
            use_intermediate=use_intermediate_for_shorty,
            mode='predict',
            flag=self.shorty is not None)
        num_instances = data_loader.dataset.num_instances
        predicted_labels = {}
        predicted_labels['knn'] = SMatrix(
            n_rows=num_instances,
            n_cols=num_labels,
            nnz=top_k)

        predicted_labels['clf'] = SMatrix(
            n_rows=num_instances,
            n_cols=num_labels,
            nnz=top_k)

        count = 0
        for batch_data in tqdm(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            self._update_predicted_shortlist(
                count, batch_size, predicted_labels,
                out_ans, batch_data)
            count += batch_size
            del batch_data
        for k, v in predicted_labels.items():
            predicted_labels[k] = v.data()
        predicted_labels['ens'] = self._combine_scores(
            predicted_labels['clf'], predicted_labels['knn'], beta)
        return predicted_labels

    def predict(self, data_dir, result_dir, dataset, data=None,
                tst_feat_fname='tst_X_Xf.txt', tst_label_fname='tst_X_Y.txt',
                lbl_feat_fname='lbl_X_Xf.txt', batch_size=256, num_workers=6,
                keep_invalid=False, feature_indices=None, label_indices=None,
                top_k=50, normalize_features=True, normalize_labels=False,
                feature_type='sparse', pretrained_shortlist=None,
                use_intermediate_for_shorty=True, beta=0.2, **kwargs):
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
        lbl_feat_fname: str, optional, default='lbl_X_Xf.txt'
            load label features from this file when data is None
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
        trn_pretrained_shortlist: csr_matrix or None, default=None
            Shortlist for test dataset
            * will directly use this this shortlist when available
        use_intermediate_for_shorty: bool
            use intermediate representation for negative sampling/ANN
        Returns
        -------
        predicted_labels: csr_matrix
            predictions for the given dataset
        """
        dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=tst_feat_fname,
            fname_labels=tst_label_fname,
            fname_label_features=lbl_feat_fname,
            data=data,
            mode='predict',
            feature_type=feature_type,
            size_shortlist=self.shortlist_size,
            _type='shortlist',
            pretrained_shortlist=pretrained_shortlist,
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices)
        data_loader = self._create_data_loader(
            feature_type=feature_type,
            classifier_type='shortlist',
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers)
        time_begin = time.time()
        predicted_labels = self._predict(
            data_loader, top_k, beta, use_intermediate_for_shorty)
        time_end = time.time()
        prediction_time = time_end - time_begin
        avg_prediction_time = prediction_time*1000/len(data_loader.dataset)
        acc = self.evaluate(dataset.labels.data, predicted_labels)
        _res = self._format_acc(acc)
        self.logger.info(
            "Prediction time (total): {:.2f} sec., "
            "Prediction time (per sample): {:.2f} msec., "
            "P@k(%): {:s}".format(
                prediction_time,
                avg_prediction_time, _res))
        return predicted_labels, prediction_time, avg_prediction_time

    def save_checkpoint(self, model_dir, epoch):
        # Avoid purge call from base class
        super().save_checkpoint(model_dir, epoch, False)
        if self.shorty is not None:
            self.tracking.saved_checkpoints[-1]['ANN'] \
                = 'checkpoint_ANN_{}.pkl'.format(epoch)
            self.shorty.save(os.path.join(
                model_dir, self.tracking.saved_checkpoints[-1]['ANN']))
        self.purge(model_dir)

    def load_checkpoint(self, model_dir, fname, epoch):
        super().load_checkpoint(model_dir, fname, epoch)
        if self.shorty is not None:
            fname = os.path.join(model_dir, 'checkpoint_ANN_{}'.format(epoch))
            self.shorty.load(fname)

    def save(self, model_dir, fname):
        super().save(model_dir, fname)
        if self.shorty is not None:
            self.shorty.save(os.path.join(model_dir, fname+'_ANN'))

    def load(self, model_dir, fname):
        super().load(model_dir, fname)
        if self.shorty is not None:
            self.shorty.load(os.path.join(model_dir, fname+'_ANN'))

    def purge(self, model_dir):
        if self.shorty is not None:
            if len(self.tracking.saved_checkpoints) \
                    > self.tracking.checkpoint_history:
                fname = self.tracking.saved_checkpoints[0]['ANN']
                self.shorty.purge(fname)  # let the class handle the deletion
        super().purge(model_dir)

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

    @property
    def model_size(self):
        s = self.net.model_size
        if self.shorty is not None:
            return s + self.shorty.model_size
        return s
