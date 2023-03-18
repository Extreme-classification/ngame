from .model_base import ModelBase
import time
import os
from .utils import get_filter_map, predict_anns
import numpy as np
import torch
from libs.batching import MySampler
from xclib.utils.sparse import csr_from_arrays, normalize
from tqdm import tqdm
import logging


def construct_model(args, net, loss, optimizer, schedular, shortlister):
    if args.network_type == 'siamese':
        if args.sampling_type == 'implicit':
            return SModelIS(
                net=net,
                criterion=loss,
                optimizer=optimizer,
                schedular=schedular,
                model_dir=args.model_dir,
                result_dir=args.result_dir)
        elif args.sampling_type == 'explicit':
            raise NotImplementedError("")
        else:
            raise NotImplementedError("")
    elif args.network_type == 'xc':
        if args.sampling_type == 'implicit':
            return XModelIS(
                net=net,
                criterion=loss,
                optimizer=optimizer,
                schedular=schedular,
                model_dir=args.model_dir,
                result_dir=args.result_dir,
                shortlister=shortlister)
        elif args.sampling_type == 'explicit':
            raise NotImplementedError("")
        else:
            raise NotImplementedError("")
    else:
        raise NotImplementedError("")


class ModelIS(ModelBase):
    """
    Generic class for models with implicit sampling
    
    Implicit sampling:
    - Negatives are not explicity sampled but selected
    from positive labels of other documents in the mini-batch
    - Also referred as in-batch or DPR in literature

    Arguments
    ---------
    net: models.network.DeepXMLBase
        * DeepXMLs: network with a label shortlist
        * DeepXMLf: network with fully-connected classifier
    criterion: libs.loss._Loss
        to compute loss given y, y_hat and mask
    optimizer: libs.optimizer.Optimizer
        to back-propagate and updating the parameters
    schedular: torch.optim.lr_schedular
        to compute loss given y, y_hat and mask
    model_dir: str
        path to model dir (will save models here)
    result_dir: str
        path to result dir (will save results here)
    result_dir: str
        path to result dir (will save results here)
    shortlister: libs.shortlist.Shortlist
        to generate a shortlist of labels (to be used at prediction time)
    """
    def __init__(
        self,
        net,
        criterion,
        optimizer,
        schedular,
        model_dir,
        result_dir,
        shortlister=None
    ):
        super().__init__(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            schedular=schedular,
            model_dir=model_dir,
            result_dir=result_dir
        )
        self.shortlister = shortlister
        self.memory_bank = None

    def _compute_loss(self, y_hat, batch_data):
        """
        Compute loss for given pair of ground truth and logits
        """
        y = batch_data['Y'].to(y_hat.device)
        mask = batch_data['Y_mask']
        return self.criterion(
            y_hat,
            y,
            mask.to(y_hat.device) if mask is not None else mask)

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

    def create_batch_sampler(self, dataset, batch_size, shuffle):
        if shuffle:
            order = dataset.indices_permutation()
        else:
            order = np.arange(len(dataset))
        return torch.utils.data.sampler.BatchSampler(
                MySampler(order), batch_size, False)

    def _strip_padding_label(self, mat, num_labels):
        stripped_vals = {}
        for key, val in mat.items():
            stripped_vals[key] = val[:, :num_labels].tocsr()
            del val
        return stripped_vals


class SModelIS(ModelIS):
    """
    For models that do Siamese training with implicit sampling
    
    * Siamese training: label embeddings are treated as classifiers
    * Implicit sampling: negatives are not explicity sampled but
    selected from positive labels of other documents in the mini-batch

    Arguments
    ---------
    net: models.network.DeepXMLBase
        * DeepXMLs: network with a label shortlist
        * DeepXMLf: network with fully-connected classifier
    criterion: libs.loss._Loss
        to compute loss given y, y_hat and mask
    optimizer: libs.optimizer.Optimizer
        to back-propagate and updating the parameters
    schedular: torch.optim.lr_schedular
        to compute loss given y, y_hat and mask
    model_dir: str
        path to model dir (will save models here)
    result_dir: str
        path to result dir (will save results here)
    shortlister: libs.shortlist.Shortlist
        to generate a shortlist of labels (to be used at prediction time)
    """
    def __init__(
        self,
        net,
        criterion,
        optimizer,
        schedular,
        model_dir,
        result_dir,
        shortlister=None
    ):
        super().__init__(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            schedular=schedular,
            model_dir=model_dir,
            result_dir=result_dir,
            shortlister=shortlister
        )

    def _fit(
        self,
        train_loader,
        validation_loader,
        init_epoch,
        num_epochs,
        validate_after,
        beta,
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
        init_epoch: int
            start training from this epoch
            (useful when fine-tuning from a checkpoint)
        num_epochs: int
            #passes over the dataset
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        beta: float
            weightage of classifier when combining with shortlist scores
        filter_map: np.ndarray or None
            mapping to filter the predictions
        sampling_params: Namespace or None
            parameters to be used for negative sampling
        """
        smp_warmup = sampling_params.curr_epochs[0]
        smp_refresh_interval = sampling_params.refresh_interval

        for epoch in range(init_epoch, init_epoch+num_epochs):
            if epoch >= smp_warmup and epoch % smp_refresh_interval == 0:
                sampling_time = time.time()
                if self.memory_bank is None:
                    _X = self.get_embeddings(
                        data=train_loader.dataset.features.data,
                        encoder=self.net.encode_document,
                        batch_size=train_loader.batch_sampler.batch_size,
                        feature_type=train_loader.dataset.feature_type
                        )
                else:
                    _X = self.memory_bank
                train_loader.dataset.update_sampler(_X)
                sampling_time = time.time() - sampling_time
                self.tracking.sampling_time += sampling_time
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
                    validation_loader.dataset.labels.data,
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

    def fit(
        self,
        data_dir,
        dataset,
        trn_fname,
        val_fname,
        trn_data=None,
        val_data=None,
        filter_file_val=None,
        feature_type='sparse',
        normalize_features=True,
        normalize_labels=False,
        sampling_params=None,
        freeze_encoder=False,
        use_amp=True,
        num_epochs=10,
        init_epoch=0,
        batch_size=128,
        num_workers=6,
        shuffle=True,
        validate=False,
        validate_after=20,
        batch_type='doc',
        max_len=32,
        beta=0.2,
        *args, **kwargs
    ):
        """
        Main training function to learn model parameters

        Arguments
        ---------
        data_dir: str or None
            load data from this directory when data is None
        dataset: str
            Name of the dataset
        trn_fname: dict
            file names to construct train dataset
            * f_features: features file
            * f_labels: labels file
            * f_label_features: label feature file
        val_fname: dict or None
            file names to construct validation dataset
            * f_features: features file
            * f_labels: labels file
            * f_label_features: label feature file
        trn_data: dict or None, optional, default=None
            directly use this this data to train when available
            * X: feature; Y: label; Yf: label_features
        val_data: dict or None, optional, default=None
            directly use this this data to validate when available
            * X: feature; Y: label; Yf: label_features
        filter_file_val: str or None
            mapping to filter the predictions for validation data
        feature_type: str, optional, default='sparse'
            sparse, sequential or dense features
        normalize_features: bool, optional, default=True
            Normalize data points to unit norm
        normalize_lables: bool, optional, default=False
            Normalize labels to convert in probabilities
            Useful in-case on non-binary labels
        sampling_params: Namespace or None
            parameters to be used for negative sampling
        freeze_encoder: boolean, optional (default=False)
            freeze the encoder (embeddings can be pre-computed for efficiency)
            * #TODO
        use_amp: boolean, optional (default=True)
            use automatic mixed-precision
        num_epochs: int
            #passes over the dataset
        init_epoch: int
            start training from this epoch
            (useful when fine-tuning from a checkpoint)
        batch_size: int, optional, default=128
            batch size in data loader
        num_workers: int, optional, default=6
            #workers in data loader
        shuffle: boolean, optional, default=True
            shuffle train data in each epoch
        validate: boolean, optional, default=True
            validate or not
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        batch_type: str, optional, default='doc'
            * doc: batch over document and sample labels
            * lbl: batch over labels and sample documents
        max_len: int, optional, default=-1
            maximum length in case of sequential features
            * -1 would keep all the tokens (trust the dumped features)
        beta: float
            weightage of classifier when combining with shortlist scores
        """
        self.setup_amp(use_amp)
        self.logger.addHandler(
            logging.FileHandler(
            os.path.join(self.result_dir, 'log_train.txt')))
        self.logger.info(f"Net: {self.net}")
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
            max_len=max_len,
            classifier_type='siamese',
            batch_type=batch_type
            )
        train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            max_len=max_len,
            feature_type=feature_type,
            classifier_type='siamese',
            sampling_type=sampling_params.type,
            num_workers=num_workers,
            shuffle=shuffle)
        if sampling_params.asynchronous:
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
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_type=feature_type,
                max_len=max_len,
                batch_type='doc'
                )
            validation_loader = self._create_data_loader(
                validation_dataset,
                feature_type=feature_type,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False)
            filter_map = get_filter_map(os.path.join(
                data_dir, dataset, filter_file_val))
        self._fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            init_epoch=init_epoch,
            num_epochs=num_epochs,
            validate_after=validate_after,
            beta=beta,
            sampling_params=sampling_params,
            filter_map=filter_map)            
        train_time = self.tracking.train_time \
            + self.tracking.shortlist_time \
            + self.tracking.sampling_time
        self.tracking.save(
            os.path.join(self.result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {:.2f} sec, Sampling time: {:.2f} sec, "
            "Shortlist time: {:.2f} sec, Validation time: {:.2f} sec, "
            "Model size: {:.2f} MB\n".format(
                self.tracking.train_time,
                self.tracking.sampling_time,
                self.tracking.shortlist_time,
                self.tracking.validation_time,
                self.model_size))
        return train_time, self.model_size

    def _validate(self, train_data_loader, data_loader, *args, **kwargs):
        self.net.eval()
        torch.set_grad_enabled(False)
        num_labels = data_loader.dataset.num_labels
        self.logger.info("Getting val document embeddings")
        val_doc_embeddings = self.get_embeddings(
            data=data_loader.dataset.features.data,
            encoder=self.net.encode_document,
            batch_size=data_loader.batch_sampler.batch_size,
            feature_type=data_loader.dataset.feature_type
            )
        self.logger.info("Getting label embeddings")
        lbl_embeddings = self.get_embeddings(
            data=train_data_loader.dataset.label_features.data,
            encoder=self.net.encode_label,
            batch_size=data_loader.batch_sampler.batch_size,
            feature_type=train_data_loader.dataset.feature_type
            )
        predicted_labels = {}
        predicted_labels['knn'] = predict_anns(
            val_doc_embeddings, lbl_embeddings)
        return self._strip_padding_label(predicted_labels, num_labels), \
            'NaN'
        

class XModelIS(ModelIS):
    """
    For models that do XC training with implicit sampling

    * XC training: classifiers and encoders (optionally) are trained    
    * Implicit sampling: negatives are not explicity sampled but
    selected from positive labels of other documents in the mini-batch

    Arguments
    ---------
    net: models.network.DeepXMLBase
        * DeepXMLs: network with a label shortlist
        * DeepXMLf: network with fully-connected classifier
    criterion: libs.loss._Loss
        to compute loss given y, y_hat and mask
    optimizer: libs.optimizer.Optimizer
        to back-propagate and updating the parameters
    schedular: torch.optim.lr_schedular
        to compute loss given y, y_hat and mask
    model_dir: str
        path to model dir (will save models here)
    result_dir: str
        path to result dir (will save results here)
    shortlister: libs.shortlist.Shortlist
        to generate a shortlist of labels (to be used at prediction time)
    """

    def __init__(
        self,
        net,
        criterion,
        optimizer,
        schedular,
        model_dir,
        result_dir,
        shortlister=None
    ):
        super().__init__(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            schedular=schedular,
            model_dir=model_dir,
            result_dir=result_dir,
            shortlister=shortlister
        )

    def _combine_scores(self, score_knn, score_clf, beta):
        """
        Combine scores of label classifier and shortlist
        score = beta*score_knn + (1-beta)*score_clf
        """
        return beta*score_knn + (1-beta)*score_clf

    def _validate(self, data_loader, beta=0.2, top_k=100):
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
        torch.set_grad_enabled(False)
        num_labels = data_loader.dataset.num_labels
        self.logger.info("Getting val document embeddings")
        val_doc_embeddings = self.get_embeddings(
            data=data_loader.dataset.features.data,
            encoder=self.net.encode_document,
            batch_size=data_loader.batch_sampler.batch_size,
            feature_type=data_loader.dataset.feature_type
            )
        self.logger.info("Getting label embeddings")
        lbl_embeddings = self.net.get_clf_weights()
        predicted_labels = {}
        predicted_labels['clf'] = predict_anns(
            val_doc_embeddings, lbl_embeddings, k=top_k)
        return self._strip_padding_label(predicted_labels, num_labels), \
            'NaN'

    def _fit(self,
             train_loader,
             validation_loader,
             init_epoch,
             num_epochs,
             validate_after,
             beta,
             sampling_params,
             filter_map):
        """
        Train for the given data loader
        Arguments
        ---------
        train_loader: DataLoader
            data loader over train dataset
        validation_loader: DataLoader or None
            data loader over validation dataset
        init_epoch: int
            start training from this epoch
            (useful when fine-tuning from a checkpoint)
        num_epochs: int
            #passes over the dataset
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        beta: float
            weightage of classifier when combining with shortlist scores
        filter_map: np.ndarray or None
            mapping to filter the predictions
        sampling_params: Namespace or None
            parameters to be used for negative sampling
        """
        smp_warmup = sampling_params.curr_epochs[0]
        smp_refresh_interval = sampling_params.refresh_interval

        for epoch in range(init_epoch, init_epoch+num_epochs):
            if epoch >= smp_warmup and epoch % smp_refresh_interval == 0:
                sampling_time = time.time()
                if self.memory_bank is None:
                    _X = self.get_embeddings(
                        data=train_loader.dataset.features.data,
                        encoder=self.net.encode_document,
                        batch_size=train_loader.batch_sampler.batch_size,
                        feature_type=train_loader.dataset.feature_type
                        )
                else:
                    _X = self.memory_bank
                train_loader.dataset.update_sampler(_X)
                sampling_time = time.time() - sampling_time
                self.tracking.sampling_time += sampling_time
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
                    validation_loader.dataset.labels.data,
                    predicted_labels, filter_map)
                self.tracking.validation_time = self.tracking.validation_time \
                    + val_end_t - val_start_t
                self.tracking.mean_val_loss.append(val_avg_loss)
                self.tracking.val_precision.append(_acc['clf'][0])
                self.tracking.val_ndcg.append(_acc['clf'][1])
                _acc = self._format_acc(_acc['clf'])
                self.logger.info("Model saved after epoch: {}".format(epoch))
                self.save_checkpoint(self.model_dir, epoch+1)
                self.tracking.last_saved_epoch = epoch
                self.logger.info(
                    "P@1 (clf): {:s}, loss: {:s},"
                    " time: {:.2f} sec".format(
                        _acc, val_avg_loss,
                        val_end_t-val_start_t))
            self.tracking.last_epoch += 1
            self.update_order(train_loader)
            train_loader.dataset.update_state()
        self.save_checkpoint(self.model_dir, epoch+1)
    
    def init_classifier(self, dataset, batch_size=128):
        lbl_embeddings = self.get_embeddings(
            data=dataset.label_features.data,
            encoder=self.net.encode_label,
            batch_size=batch_size,
            feature_type=dataset.feature_type
            )
        self.net.initialize_classifier(
            np.vstack(
            [normalize(lbl_embeddings),
             np.zeros((1, self.net.representation_dims))])
            )

    def fit(self,
            data_dir,
            dataset,
            trn_fname,
            val_fname,
            trn_data=None,
            val_data=None,
            filter_file_val=None,
            feature_type='sparse',
            normalize_features=True,
            normalize_labels=False,
            sampling_params=None,
            freeze_encoder=False,
            use_amp=True,
            num_epochs=10,
            init_epoch=0,
            batch_size=128,
            num_workers=6,
            shuffle=False,
            validate=False,
            validate_after=5,
            batch_type='doc',
            max_len=-1,
            beta=0.2,
            *args, **kwargs):
        """
        Main training function to learn model parameters

        Arguments
        ---------
        data_dir: str or None
            load data from this directory when data is None
        dataset: str
            Name of the dataset
        trn_fname: dict
            file names to construct train dataset
            * f_features: features file
            * f_labels: labels file
            * f_label_features: label feature file
        val_fname: dict or None
            file names to construct validation dataset
            * f_features: features file
            * f_labels: labels file
            * f_label_features: label feature file
        trn_data: dict or None, optional, default=None
            directly use this this data to train when available
            * X: feature; Y: label; Yf: label_features
        val_data: dict or None, optional, default=None
            directly use this this data to validate when available
            * X: feature; Y: label; Yf: label_features
        filter_file_val: str or None
            mapping to filter the predictions for validation data
        feature_type: str, optional, default='sparse'
            sparse, sequential or dense features
        normalize_features: bool, optional, default=True
            Normalize data points to unit norm
        normalize_lables: bool, optional, default=False
            Normalize labels to convert in probabilities
            Useful in-case on non-binary labels
        sampling_params: Namespace or None
            parameters to be used for negative sampling
        freeze_encoder: boolean, optional (default=False)
            freeze the encoder (embeddings can be pre-computed for efficiency)
            * #TODO
        use_amp: boolean, optional (default=True)
            use automatic mixed-precision
        num_epochs: int
            #passes over the dataset
        init_epoch: int
            start training from this epoch
            (useful when fine-tuning from a checkpoint)
        batch_size: int, optional, default=128
            batch size in data loader
        num_workers: int, optional, default=6
            #workers in data loader
        shuffle: boolean, optional, default=True
            shuffle train data in each epoch
        validate: boolean, optional, default=True
            validate or not
        validate_after: int, optional, default=5
            validate after a gap of these many epochs
        batch_type: str, optional, default='doc'
            * doc: batch over document and sample labels
            * lbl: batch over labels and sample documents
        max_len: int, optional, default=-1
            maximum length in case of sequential features
            * -1 would keep all the tokens (trust the dumped features)
        beta: float
            weightage of classifier when combining with shortlist scores
        """
        self.setup_amp(use_amp)
        # Reset the logger to dump in train log file
        self.logger.addHandler(
            logging.FileHandler(
            os.path.join(self.result_dir, 'log_train.txt')))
        self.logger.info(f"Net: {self.net}")
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
            max_len=max_len,
            classifier_type='xc',
            batch_type=batch_type
            )
        _train_dataset = train_dataset
        self.init_classifier(train_dataset)

        train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            prefetch_factor=5,
            feature_type=feature_type,
            num_workers=num_workers,
            classifier_type='xc',
            sampling_type=sampling_params.type,
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
                os.path.join(data_dir, dataset),
                fname=val_fname,
                data=val_data,
                mode='predict',
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_type=feature_type,
                max_len=max_len,
                batch_type='doc'
                )
            validation_loader = self._create_data_loader(
                validation_dataset,
                batch_size=batch_size,
                feature_type=feature_type,
                prefetch_factor=5,
                num_workers=num_workers,
                shuffle=False)
            filter_map = get_filter_map(os.path.join(
                data_dir, dataset, filter_file_val))
        del _train_dataset
        self._fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            init_epoch=init_epoch,
            num_epochs=num_epochs,
            validate_after=validate_after,
            beta=beta,
            sampling_params=sampling_params,
            filter_map=filter_map)

        # learn anns over label embeddings and label classifiers
        self.post_process_for_inference(train_dataset, batch_size)
        train_time = self.tracking.train_time \
            + self.tracking.shortlist_time \
            + self.tracking.sampling_time
        self.tracking.save(
            os.path.join(self.result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {:.2f} sec, Sampling time: {:.2f} sec, "
            "Shortlist time: {:.2f} sec, Validation time: {:.2f} sec, "
            "Model size: {:.2f} MB\n".format(
                self.tracking.train_time,
                self.tracking.sampling_time,
                self.tracking.shortlist_time,
                self.tracking.validation_time,
                self.model_size))
        return train_time, self.model_size

    def post_process_for_inference(self, dataset, batch_size=128):
        start_time = time.time()
        self.net.eval()
        torch.set_grad_enabled(False)
        embeddings = self.get_embeddings(
            data=dataset.label_features.data,
            encoder=self.net.encode_label,
            batch_size=batch_size,
            feature_type=dataset.feature_type
            )
        classifiers = self.net.get_clf_weights()
        self.logger.info("Training ANNS..")
        self.shortlister.fit(embeddings, classifiers)
        self.tracking.shortlist_time += (time.time() - start_time) 

    def _predict(self, dataset, top_k, batch_size, num_workers, 
                 beta):
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

        self.logger.info("Getting test embeddings.")
        num_instances = dataset.num_instances
        num_labels = dataset.num_labels

        doc_embeddings = self.get_embeddings(
            data=dataset.features.data,
            encoder=self.net.encode_document,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_type=dataset.feature_type
            )

        self.logger.info("Querying ANNS..")
        predicted_labels = {}
        pred_knn, pred_clf = self.shortlister.query(doc_embeddings)
        predicted_labels['knn'] =  csr_from_arrays(
            pred_knn[0], pred_knn[1],
            shape=(num_instances, num_labels+1)
            )[:, :-1]

        predicted_labels['clf'] =  csr_from_arrays(
            pred_clf[0], pred_clf[1],
            shape=(num_instances, num_labels+1)
            )[:, :-1]

        predicted_labels['ens'] = self._combine_scores(
            predicted_labels['clf'], predicted_labels['knn'], beta)
        return predicted_labels

    def predict(self, 
                data_dir,
                result_dir,
                dataset,
                fname,
                data=None,
                max_len=32,
                batch_size=128,
                num_workers=4,
                normalize_features=True,
                normalize_labels=False,
                beta=0.2,
                top_k=100,
                feature_type='sparse',
                filter_map=None,
                **kwargs):
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
        self.logger.addHandler(
            logging.FileHandler(os.path.join(result_dir, 'log_predict.txt')))
        filter_map = np.loadtxt(os.path.join(
                data_dir, dataset, filter_map)).astype(np.int)
        dataset = self._create_dataset(
            data_dir=os.path.join(data_dir, dataset),
            fname=fname,
            data=data,
            mode='test',
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_type=feature_type,
            max_len=max_len,
            batch_type='doc'
            )
        time_begin = time.time()
        predicted_labels = self._predict(
            dataset, top_k, batch_size, num_workers, beta)
        time_end = time.time()
        prediction_time = time_end - time_begin
        avg_prediction_time = prediction_time*1000/len(dataset)
        acc = self.evaluate(dataset.labels.data, predicted_labels, filter_map)
        _res = self._format_acc(acc)
        self.logger.info(
            "Prediction time (total): {:.2f} sec., "
            "Prediction time (per sample): {:.2f} msec., "
            "P@k(%): {:s}".format(
                prediction_time,
                avg_prediction_time, _res))
        return predicted_labels, prediction_time, avg_prediction_time

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
        super().load(model_dir, fname)
        if self.shortlister is not None:
            self.shortlister.load(os.path.join(model_dir, fname+'_ANN'))

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
        super().load(model_dir, fname)
        if self.shortlister is not None:
            self.shortlister.load(os.path.join(model_dir, fname+'_ANN'))

    @property
    def model_size(self):
        s = self.net.model_size
        if self.shortlister is not None:
            return s + self.shortlister.model_size
        return s
