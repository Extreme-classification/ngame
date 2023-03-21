dataset=$1
max_len=$2

work_dir=$(cd ../../../../ && pwd)
data_dir="${work_dir}/data/${dataset}"

# extract raw-text
echo -e "Extracting raw text..\n"

python3 -W ignore ../tools/extract_raw_text.py "${data_dir}/trn.json.gz" "${data_dir}/trn.raw.txt"
python3 -W ignore ../tools/extract_raw_text.py "${data_dir}/tst.json.gz" "${data_dir}/tst.raw.txt"
python3 -W ignore ../tools/extract_raw_text.py "${data_dir}/lbl.json.gz" "${data_dir}/lbl.raw.txt"

# tokenize
echo -e "Tokenizing..\n"
python3 -W ignore ../tools/create_tokenized_files.py --data-dir "${data_dir}" --max-length ${max_len} --tokenizer-type "bert-base-uncased"
