from xclib.data.data_utils import read_corpus
import sys
import os


def main(in_fname, op_fname, fields):
    with open(op_fname, 'w', encoding='latin') as fp:
        for line in read_corpus(in_fname):
            t = ""
            for f in fields:
                t += f"{line[f]} "
            fp.write(t.strip() + "\n")


if __name__ == "__main__":
    in_fname = sys.argv[1]
    op_fname = sys.argv[2]

    dataset = os.path.split(
        os.path.dirname(in_fname))[1]

    if 'title' in dataset.lower():
        fields = ["title"]
    else:
        fields = ["title", "content"]

    main(in_fname, op_fname, fields)
