dataset=$1
max_len=$2

work_dir=$(cd ../../../../ && pwd)
data_dir="${work_dir}/data/${dataset}"

# extract raw-text
echo -e "Extracting raw text and labels..\n"

python3 -W ignore ../tools/extract_text_and_labels.py "${data_dir}" 

# tokenize
echo -e "Tokenizing..\n"
python3 -W ignore ../tools/create_tokenized_files.py --data-dir "${data_dir}" --max-length ${max_len} --tokenizer-type "bert-base-uncased"
