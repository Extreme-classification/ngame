import sys
import os
from xclib.data.data_utils import read_corpus
from xclib.utils.sparse import ll_to_sparse
from xclib.data import data_utils


def process_text(in_fname, op_fname, fields):
    count = 0
    with open(op_fname, 'w', encoding='latin') as fp:
        for line in read_corpus(in_fname):
            t = ""
            for f in fields:
                t += f"{line[f]} "
            fp.write(t.strip() + "\n")
            count += 1
    return count


def process_text_labels(in_fname, op_tfname, op_lfname, fields, num_labels=-1):
    labels = []
    with open(op_tfname, 'w', encoding='latin') as fp:
        for line in read_corpus(in_fname):
            t = ""
            labels.append(line['target_ind'])
            for f in fields:
                t += f"{line[f]} "
            fp.write(t.strip() + "\n")
    if num_labels == -1:
        max_ind = max([max(item) for item in labels])
        print("num_labels is -1; will be determining index from json.gz")
        num_labels = max_ind
    labels = ll_to_sparse(
        labels, shape=(len(labels), num_labels))
    data_utils.write_sparse_file(labels, op_lfname, header=True)


def main():
    data_dir = sys.argv[1]    
    dataset = os.path.split(data_dir)[1]

    if 'title' in dataset.lower():
        fields = ["title"]
    else:
        fields = ["title", "content"]

    # num_labels can be helpful when last few labels 
    # appears either in train and test
    num_labels = process_text(
        in_fname=os.path.join(data_dir, 'lbl.json.gz'),
        op_fname=os.path.join(data_dir, 'lbl.raw.txt'),
        fields=fields
    )

    process_text_labels(
        in_fname=os.path.join(data_dir, 'trn.json.gz'),
        op_tfname=os.path.join(data_dir, 'trn.raw.txt'),
        op_lfname=os.path.join(data_dir, 'trn_X_Y.txt'),
        fields=fields,
        num_labels=num_labels
    )

    process_text_labels(
        in_fname=os.path.join(data_dir, 'tst.json.gz'),
        op_tfname=os.path.join(data_dir, 'tst.raw.txt'),
        op_lfname=os.path.join(data_dir, 'tst_X_Y.txt'),
        fields=fields,
        num_labels=num_labels
    )


if __name__ == "__main__":
    main()
