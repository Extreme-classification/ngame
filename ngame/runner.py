import libs.parameters as parameters
import json
import sys
import os
from main import main
import shutil
import tools.evaluate as evalaute
import tools.surrogate_mapping as surrogate_mapping


def create_surrogate_mapping(data_dir, g_config, seed):
    """
    In case of SiameseXML: it'll just remove invalid labels
    However, keeping this code as user might want to try out 
    alternate mappings as well
    
    ##FIXME: For non-shared vocabulary
    """
    dataset = g_config['dataset']
    try:
        surrogate_threshold = g_config['surrogate_threshold']
        surrogate_method = g_config['surrogate_method']
    except KeyError:
        surrogate_threshold = -1
        surrogate_method = 0
    
    arch = g_config['arch']
    tmp_model_dir = os.path.join(
        data_dir, dataset, f'siamesexml.{arch}', f"{surrogate_threshold}.{seed}")
    data_dir = os.path.join(data_dir, dataset)
    try:
        os.makedirs(tmp_model_dir, exist_ok=False)
        surrogate_mapping.run(
            feat_fname=None,
            lbl_feat_fname=None,
            lbl_fname=os.path.join(data_dir, g_config["trn_label_fname"]),
            feature_type=g_config["feature_type"],
            method=surrogate_method,
            threshold=surrogate_threshold,
            seed=seed,
            tmp_dir=tmp_model_dir)
    except FileExistsError:
        print("Using existing data for surrogate task!")
    finally:
        data_stats = json.load(
            open(os.path.join(tmp_model_dir, "data_stats.json")))
        mapping = os.path.join(
            tmp_model_dir, 'surrogate_mapping.txt')
    return data_stats, mapping


def evaluate(config, data_dir, pred_fname, trn_pred_fname=None, n_learners=1):
    if n_learners == 1:
        # use score-fusion to combine and then evaluate
        if config["eval_method"] == "score_fusion":
            func = evalaute.eval_with_score_fusion
        # predictions are either from embedding or classifier
        elif config["inference_method"] == "traditional":
            func = evalaute.eval
        else: 
            raise NotImplementedError("")
    else:
        raise NotImplementedError("")

    data_dir = os.path.join(data_dir, config['dataset'])
    ans = func(
        tst_label_fname=os.path.join(
            data_dir, config["tst_label_fname"]),
        trn_label_fname=os.path.join(
            data_dir, config["trn_label_fname"]),
        pred_fname=pred_fname,
        trn_pred_fname=trn_pred_fname,
        A=config['A'], 
        B=config['B'],
        filter_fname=os.path.join(
            data_dir, config["tst_filter_fname"]), 
        trn_filter_fname=os.path.join(
            data_dir, config["trn_filter_fname"]),
        beta=config['beta'], 
        top_k=config['top_k'],
        save=config["save_predictions"])
    return ans


def print_run_stats(train_time, model_size, avg_prediction_time, fname=None):
    line = "-"*30 
    out = f"Training time (sec): {train_time:.2f}\n"
    out += f"Model size (MB): {model_size:.2f}\n"
    out += f"Avg. Prediction time (msec): {avg_prediction_time:.2f}"
    out = f"\n\n{line}\n{out}\n{line}\n\n"
    print(out)
    if fname is not None:
        with open(fname, "a") as fp:
            fp.write(out)


def run_ngame(work_dir, pipeline, version, seed, config):

    # fetch arguments/parameters like dataset name, A, B etc.
    g_config = config['global']
    dataset = g_config['dataset']
    arch = g_config['arch']

    # run stats
    train_time = 0
    model_size = 0
    avg_prediction_time = 0

    # Directory and filenames
    data_dir = os.path.join(work_dir, 'data')
    
    result_dir = os.path.join(
        work_dir, 'results', pipeline, arch, dataset, f'v_{version}')
    model_dir = os.path.join(
        work_dir, 'models', pipeline, arch, dataset, f'v_{version}')
    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['siamese'])
    _args.params.seed = seed

    args = _args.params
    args.data_dir = data_dir
    args.model_dir = os.path.join(model_dir, 'siamese')
    args.result_dir = os.path.join(result_dir, 'siamese')

    # Create the label mapping for classification surrogate task
    data_stats, args.surrogate_mapping = create_surrogate_mapping(
        data_dir, g_config, seed)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # train intermediate representation
    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['surrogate'].split(",")
    args.num_labels = int(temp[2])

    ##FIXME: For non-shared vocabulary

    _train_time, _ = main(args)
    train_time += _train_time

    # set up things to train extreme classifiers
    _args.update(config['extreme'])
    args = _args.params
    args.surrogate_mapping = None
    args.model_dir = os.path.join(model_dir, 'extreme')
    args.result_dir = os.path.join(result_dir, 'extreme')
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # train extreme classifiers
    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])
    _train_time, _model_size = main(args)
    train_time += _train_time
    model_size += _model_size

    # predict using shortlist and extreme classifiers for test set
    args.pred_fname = 'tst_predictions'
    args.filter_map = g_config["tst_filter_fname"]
    args.mode = 'predict'
    _, _, _pred_time = main(args)
    avg_prediction_time += _pred_time
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_clf.npz'),
        os.path.join(result_dir, 'tst_predictions_clf.npz'))
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_knn.npz'),
        os.path.join(result_dir, 'tst_predictions_knn.npz'))

    if config['extreme']["inference_method"] == "dual_mips":
        # predict using extreme classifiers and shortlist for train set
        # required for score fusion 
        # (validation set can be used here, if available)
        args.pred_fname = 'trn_predictions'
        args.filter_map = g_config["trn_filter_fname"]
        args.mode = 'predict'
        args.tst_feat_fname = g_config["trn_feat_fname"]
        args.tst_label_fname = g_config["trn_label_fname"]
        _, _, _pred_time = main(args)

        #copy the prediction files to level-1
        shutil.copy(
            os.path.join(result_dir, 'extreme', 'trn_predictions_clf.npz'),
            os.path.join(result_dir, 'trn_predictions_clf.npz'))
        shutil.copy(
            os.path.join(result_dir, 'extreme', 'trn_predictions_knn.npz'),
            os.path.join(result_dir, 'trn_predictions_knn.npz'))

        # evaluate
        ans = evaluate(
            config=g_config,
            data_dir=data_dir,
            pred_fname=os.path.join(result_dir, 'tst_predictions'),
            trn_pred_fname=os.path.join(result_dir, 'trn_predictions'),
            )
    else:
        # evaluate
        ans = evaluate(
            config=g_config,
            data_dir=data_dir,
            pred_fname=os.path.join(result_dir, 'tst_predictions'),
            )

    print(ans)
    f_rstats = os.path.join(result_dir, 'log_eval.txt')
    with open(f_rstats, "w") as fp:
        fp.write(ans)

    print_run_stats(train_time, model_size, avg_prediction_time, f_rstats)
    return os.path.join(result_dir, f"score_{g_config['beta']:.2f}.npz"), \
        train_time, model_size, avg_prediction_time


if __name__ == "__main__":
    pipeline = sys.argv[1]
    work_dir = sys.argv[2]
    version = sys.argv[3]
    config = sys.argv[4]
    seed = int(sys.argv[5])
    if pipeline == "NGAME" or pipeline == "SiameseXML++":
        run_ngame(
            pipeline=pipeline,
            work_dir=work_dir,
            version=f"{version}_{seed}",
            seed=seed,
            config=json.load(open(config)))
    else:
        raise NotImplementedError("")
