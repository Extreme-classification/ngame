import os
import argparse
import numpy as np
import torch
from models.network import Network
from libs.model import ModelSiamese
import libs.loss as loss


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
    return torch.optim.AdamW(
        net.parameters(),
        lr=params.learning_rate,
        eps=1e-06,
        weight_decay=params.weight_decay)


def construct_schedular(params, optimizer):
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=params.num_epochs*(params.num_points/params.batch_size))


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
        print("Using triplet loss, with margin: ", params.margin)
        return loss.TripletMarginLossOHNM(
            reduction=_reduction,
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
        args.data_dir,
        args.dataset,
        trn_fname,
        val_fname,
        batch_type='doc',
        validate=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size)
    
    return output


def main(args):
    net = Network('msmarco-distilbert-base-v4')
    net.to("cuda")
    model_dir = args.model_dir
    result_dir = args.result_dir
    if args.mode == 'train':
        loss = construct_loss(args)
        optimizer = construct_optimizer(args, net)
        schedular = construct_schedular(args, optimizer)
        model = ModelSiamese(net, loss, optimizer, schedular, model_dir, result_dir, feature_type=args.feature_type)
        output = train(model, args)
    elif args.mode == 'predict':
        pass
    else:
        raise NotImplementedError("")
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--num_epochs", type=int, help="The number of epochs to run for", default=300)
    parser.add_argument("--num_steps", type=int, help="The number of data points", default=294805)
    parser.add_argument("--batch_size", type=int, help="The batch size", default=1600)
    parser.add_argument("--margin", type=float, help="Margin below which negative labels are not penalized", default=0.3)
    parser.add_argument("-A", type=float, help="The propensity factor A" , default=0.55)
    parser.add_argument("-B", type=float, help="The propensity factor B", default=1.5)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.0002)
    parser.add_argument("--momentum", type=float, help="learning rate", default=0)
    parser.add_argument("--weight_decay", type=float, help="learning rate", default=0.01)
    parser.add_argument("--save-model", type=int, help="Should the model be saved", default=0)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory path - with {trn,tst}_X.txt, {trn,tst}_X_Y.txt and Y.txt")
    parser.add_argument("--file-type", type=str, required=False, help="File type txt/npz", default="txt")
    parser.add_argument("--version", type=str, help="Version of the run", default="0")
    parser.add_argument("--filter-labels-test", type=int, help="Whether to filter labels at validation time", default=1)
    parser.add_argument("--eval-interval", type=int, help="The numbers of epochs between acc evalulation", default=30)
    parser.add_argument("--loss", type=str, help="Squared or sqrt", default='triplet_margin_ohnm')
    parser.add_argument("--max-length", type=int, help="Max length for tokenizer", default=32)
    parser.add_argument("--k", type=int, help="Number of negatives to use", default=5)
    parser.add_argument("--num-negatives", type=int, help="Number of negatives to use", default=3)
    parser.add_argument("--tokenizer-type", type=str, help="Tokenizer to use", default="bert-base-uncased")
    parser.add_argument("--encoder-name", type=str, help="Encoder to use", default="msmarco-distilbert-base-v3")
    parser.add_argument("--transform-dim", type=int, help="Transform bert embeddings to size", default=-1)


    parser.add_argument("--cl-size", type=int, help="cluster size", default=32)
    parser.add_argument("--cl-start", type=int, help="", default=999999)
    parser.add_argument("--cl-update", type=int, help="", default=5)



    args = parser.parse_args()
    main(args)
