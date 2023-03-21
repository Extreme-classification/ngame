import os
import libs.utils as utils
from models.network import construct_network
from libs.model import construct_model
from libs.shortlist import construct_shortlister
from libs.optim import construct_schedular, construct_optimizer
from libs.loss import construct_loss


def train(model, args):
    """Train the model with given data
    Arguments
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    trn_fname = {
        'f_features': args.trn_feat_fname,
        'f_label_features': args.lbl_feat_fname,
        'f_labels': args.trn_label_fname}
    val_fname = {
        'f_features': args.val_feat_fname,
        'f_label_features': args.lbl_feat_fname,
        'f_labels': args.val_label_fname}

    output = model.fit(
        data_dir=args.data_dir,
        dataset=args.dataset,
        trn_fname=trn_fname,
        val_fname=val_fname,
        batch_type='doc',
        validate=True,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        sampling_params=utils.filter_params(args, 'sampling_'),
        max_len=args.max_length,
        feature_type=args.feature_type,
        label_type='sparse',
        use_amp=args.use_amp,
        freeze_encoder=args.freeze_encoder,
        validate_after=args.validate_after,
        filter_file_val=args.val_filter_fname,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size)
    model.save(args.model_dir, args.model_fname)
    return output


def inference(model, args):
    """Predict the top-k labels for given test data
    Arguments
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    fname = {
        'f_features': args.tst_feat_fname,
        'f_label_features': args.lbl_feat_fname,
        'f_labels': args.tst_label_fname}

    predicted_labels, prediction_time, avg_prediction_time = model.predict(
        data_dir=args.data_dir,
        dataset=args.dataset,
        fname=fname,
        feature_type=args.feature_type,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        max_len=args.max_length,
        filter_map=args.filter_map,
        batch_size=args.batch_size*4)
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


def initialize(args, net):
    if args.init == 'intermediate':
        print("Loading intermediate representation.")
        net.load_intermediate_model(
            os.path.join(os.path.dirname(args.model_dir), "Z.pkl"))
    elif args.init == 'token_embeddings':
        print("Loading pre-trained token embeddings.")
        embeddings = utils.load_token_emeddings(
            data_dir=os.path.join(args.data_dir, args.dataset),
            embeddings=args.token_embeddings,
            feature_indices=args.feature_indices)
        net.initialize(embeddings)
        del embeddings
    elif args.init == 'auto':
        print("Automatic initialization.")
    else:  # trust the random init
        print("Random initialization.")
    return net


def main(args):
    args.label_padding_index = args.num_labels
    net = construct_network(args)
    initialize(args, net)
    print(f"\nModel parameters: {args}\n")
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
