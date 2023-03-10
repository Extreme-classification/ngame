import argparse
import torch
from models.network import DeepXMLSS, SiameseXML
from libs.model import ModelSiamese, ModelSShortlist
import libs.loss as loss
import libs.shortlist as shortlist
import os
import libs.utils as utils


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def construct_optimizer(params, net):
    if params.optim == 'Adam':
        return torch.optim.SparseAdam(
            list(net.classifier.parameters()),
            lr=params.learning_rate,
            eps=1e-06)
    elif params.optim == 'AdamW':
        return torch.optim.AdamW(
            net.parameters(),
            lr=params.learning_rate,
            eps=1e-06,
            weight_decay=params.weight_decay)
    else:
        raise NotImplementedError("")


def construct_schedular(params, optimizer):
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=params.warmup_steps,
        num_training_steps=params.num_epochs*(
            params.num_points/params.batch_size))


def construct_loss(params, pos_weight=1.0):
    """
    Return the loss
    Arguments:
    ----------
    params: NameSpace
        parameters of the model
        * mean: mean over all entries/terms (used with OVA setting)
        * sum: sum over all entries/terms (used with a shortlist)
               - the loss is then divided by batch_size resulting in
                 sum over labels and mean over data-points in a batch
    pos_weight: int or None, optional, default=None
        weight the loss terms where y_nl = 1
    """
    _reduction = 'mean'
    # pad index is for OVA training and not shortlist
    # pass mask for shortlist
    _pad_ind = None #if params.use_shortlist else params.label_padding_index
    if params.loss == 'bce':
        return loss.BCEWithLogitsLoss(
            reduction=_reduction,
            pad_ind=_pad_ind,
            pos_weight=None)
    elif params.loss == 'triplet_margin_ohnm':
        return loss.TripletMarginLossOHNM(
            reduction=_reduction,
            apply_softmax=params.loss_agressive,
            tau=0.1,
            k=params.loss_num_negatives,
            margin=params.margin)
    elif params.loss == 'hinge_contrastive':
        return loss.HingeContrastiveLoss(
            reduction=_reduction,
            pos_weight=pos_weight,
            margin=params.margin)
    elif params.loss == 'prob_contrastive':
        return loss.ProbContrastiveLoss(
            reduction=_reduction,
            c=0.75,
            d=3.0,
            pos_weight=pos_weight,
            threshold=params.margin)
    elif params.loss == 'kprob_contrastive':
        return loss.kProbContrastiveLoss(
            k=params.k,
            reduction='custom',
            c=0.9,
            d=1.5,
            apply_softmax=False,
            pos_weight=pos_weight)


def train(model, args):
    trn_fname = {
        'feature_fname': args.trn_feat_fname,
        'label_feature_fname': args.lbl_feat_fname,
        'label_fname': args.trn_label_fname}
    val_fname = {
        'feature_fname': args.val_feat_fname,
        'label_feature_fname': args.lbl_feat_fname,
        'label_fname': args.val_label_fname}

    output = model.fit(
        data_dir=args.data_dir,
        dataset=args.dataset,
        trn_fname=trn_fname,
        val_fname=val_fname,
        batch_type='doc',
        validate=True,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        sampling_params=args,
        max_len=args.max_length,
        validate_after=args.validate_after,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size)
    model.save(args.model_dir, args.model_fname)
    return output


def inference(model, args):
    fname = {
        'feature_fname': args.tst_feat_fname,
        'label_feature_fname': args.lbl_feat_fname,
        'label_fname': args.tst_label_fname}

    predicted_labels, prediction_time, avg_prediction_time = model.predict(
        data_dir=args.data_dir,
        dataset=args.dataset,
        fname=fname,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        max_len=args.max_length,
        filter_map=args.filter_map,
        batch_size=args.batch_size)
    model.save(args.model_dir, args.model_fname)
    utils.save_predictions(
        preds=predicted_labels,
        result_dir=args.result_dir,
        valid_labels=None,
        num_samples=None,
        num_labels=None,
        prefix=args.pred_fname,
        get_fnames=["knn", "clf"])
    return predicted_labels, prediction_time, avg_prediction_time


def construct_network(args):
    if args.network_type == 'siamese':
        net = SiameseXML(args)
    elif args.network_type == 'sshortlist':
        print("With sshortlist...")
        net = DeepXMLSS(args)
    else:
        raise NotImplementedError("")

    if args.init == 'intermediate':
        print("Loading intermediate representation.")
        net.load_intermediate_model(
            os.path.join(os.path.dirname(args.model_dir), "Z.pkl"))
    elif args.init == 'token_embeddings':
        print("Loading pre-trained token embeddings.")
        embeddings = load_emeddings(args)
        net.initialize(embeddings)
        del embeddings
    elif args.init == 'auto':
        print("Automatic initialization.")
    else:  # trust the random init
        print("Random initialization.")
    return net


def construct_model(args, net, loss, optimizer, schedular, shortlister):
    if args.model_type == 'siamese':
        return ModelSiamese(
            net=net,
            criterion=loss,
            optimizer=optimizer,
            schedular=schedular,
            model_dir=args.model_dir,
            result_dir=args.result_dir,
            feature_type=args.feature_type,
            use_amp=args.use_amp)
    elif args.model_type == 'sshortlist':
        return ModelSShortlist(
            net=net,
            criterion=loss,
            optimizer=optimizer,
            schedular=schedular,
            model_dir=args.model_dir,
            result_dir=args.result_dir,
            feature_type=args.feature_type,
            shortlister=shortlister,
            use_amp=args.use_amp)
    else:
        raise NotImplementedError("")


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
        shortlister = shortlist.ShortlistMIPS(
            method=args.ann_method,
            num_neighbours=args.num_nbrs,
            M=args.M,
            efC=args.efC,
            efS=args.efS,
            num_threads=args.ann_threads)
    elif args.inference_method == 'dual_mips':
        shortlister = shortlist.DualShortlistMIPS(
            method=args.ann_method,
            num_neighbours=args.num_nbrs,
            M=args.M,
            efC=args.efC,
            efS=args.efS,
            num_threads=args.ann_threads)
    else:
        shortlister = None
    return shortlister


def main(args):
    args.label_padding_index = args.num_labels
    net = construct_network(args)
    print("Model parameters: ", args)
    net.to("cuda")
    shortlister = construct_shortlister(args)
    if args.mode == 'train':
        loss = construct_loss(args)
        optimizer = construct_optimizer(args, net)
        schedular = construct_schedular(args, optimizer)
        model = construct_model(
            args, net, loss, optimizer, schedular, shortlister)
        output = train(model, args)
        if args.save_intermediate:
            net.save_intermediate_model(
                os.path.join(os.path.dirname(args.model_dir), "Z.pkl"))
    elif args.mode == 'predict':
        model = construct_model(args, net, None, None, None, shortlister)
        model.load(args.model_dir, args.model_fname)
        output = inference(model, args)
    elif args.mode == 'encode':
        pass
    else:
        raise NotImplementedError("")
    return output


if __name__ == "__main__":
    pass
