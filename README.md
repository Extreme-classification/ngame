# ngame

Code for _NGAME: Negative mining-aware mini-batching for extreme classification_ [1]

---

## Setting up

---

### Expected directory structure

```txt
+-- <work_dir>
|  +-- programs
|  |  +-- ngame
|  |    +-- ngame
|  +-- data
|    +-- <dataset>
|  +-- models
|  +-- results
```

### Download data for NGAME

```txt
* Download the (zipped file) BoW features and raw data from The XML repository [5].  
* Extract the zipped file into data directory. 
* The following files should be available in <work_dir>/data/<dataset> for new datasets (ignore the next step)
    - trn.json.gz
    - trn_X_Y.txt
    - tst.json.gz
    - tst_X_Y.txt
    - tst.json.gz
    - filter_labels_text.txt
* The following files would be available in <work_dir>/data/<dataset> if the dataset is in old format (please refer to next step to convert the data to new format)
    - train.txt
    - test.txt
```

### Convert to new data format (optional)

```perl
# A perl script is provided (in ngame/tools) to convert the data into new format
# Either set the $data_dir variable to the data directory of a particular dataset or replace it with the path
perl convert_format.pl $data_dir/train.txt $data_dir/trn_X_Xf.txt $data_dir/trn_X_Y.txt
perl convert_format.pl $data_dir/test.txt $data_dir/tst_X_Xf.txt $data_dir/tst_X_Y.txt
```

## Example use cases

---

### A single learner

Tokenize data as follows.

```bash
./prepare_data.sh LF-AmazonTitles-131K 32
```

The algorithm can be run as follows. A json file (e.g., config/NGAME/LF-AmazonTitles-131K.json) is used to specify architecture and other arguments. Please refer to the full documentation below for more details.

```bash
./run_main.sh 0 NGAME LF-AmazonTitles-131K 0 108
```

## Full Documentation

### Tokenize the data

```txt
./prepare_data.sh <dataset> <seq-len>

* dataset
  - Name of the dataset.
  - Tokenizer expects the following files in <work_dir>/data/<dataset>
    - trn.json.gz
    - tst.json.gz
    - tst.json.gz
  - it'll dump the following six tokenized files
    - trn_doc_input_ids.npy
    - trn_doc_attention_mask.npy
    - tst_doc_input_ids.npy
    - tst_doc_attention_mask.npy
    - lbl_input_ids.npy
    - lbl_attention_mask.npy


* seq-len
  - sequence length of text to consider while tokenizing
  - 32 for titles dataset
  - 256 for Wikipedia
  - 128 for other full-text datasets
```

### Run NGAME

```txt
./run_main.sh <gpu_id> <type> <dataset> <version> <seed>

* gpu_id: Run the program on this GPU.

* type
  NGAME builds upon SiameseXML [2] and DeepXML[3] for training. An encoder is trained in M1 and the classifier is trained in M-IV.
  - NGAME: The intermediate representation is not fine-tuned while training the classifier (more scalable; suitable for large datasets).
  - NGAME++: The intermediate representation is fine-tuned while training the classifier (leads to better accuracy on some datasets). #TODO

* dataset
  - Name of the dataset.
  - NGAME expects the following files in <work_dir>/data/<dataset>
    - trn_doc_input_ids.npy
    - trn_doc_attention_mask.npy
    - trn_X_Y.txt
    - tst_doc_input_ids.npy
    - tst_doc_attention_mask.npy
    - tst_X_Y.txt
    - lbl_input_ids.npy
    - lbl_attention_mask.npy
    - filter_labels_test.txt (put empty file or set as null in config when unavailable)

* version
  - different runs could be managed by version and seed.
  - models and results are stored with this argument.

* seed
  - seed value as used by numpy and PyTorch.
```

## TODO

- [x] Training encoders
- [x] Training classifiers
- [ ] Getting embeddings
- [x] Prediction
- [x] Score-fusion
- [ ] Other feature encoders
- [ ] Non-shared shortlist
- [ ] Other negative-samplers
- [x] tokenizer
- [ ] Multi GPU Training

## Cite as

```bib
@InProceedings{Dahiya23,
    author = "Dahiya, K. and Gupta, N. and Saini, D. and Soni, A. and Wang, Y. and Dave, K. and Jiao, J. and Gururaj, K. and Dey, P. and Singh, A. and Hada, D. and Jain, V. and Paliwal, B. and Mittal, A. and Mehta, S. and Ramjee, R. and Agarwal, S. and Kar, P. and Varma, M.",
    title = "NGAME: Negative mining-aware mini-batching for extreme classification",
    booktitle = "WSDM",
    month = "March",
    year = "2023"
}
```

## YOU MAY ALSO LIKE

- [SiameseXML: Siamese networks meet extreme classifiers with 100M labels](https://github.com/Extreme-classification/siamesexml)
- [DeepXML: A Deep Extreme Multi-Label Learning Framework Applied to Short Text Documents](https://github.com/Extreme-classification/deepxml)
- [DECAF: Deep Extreme Classification with Label Features](https://github.com/Extreme-classification/DECAF)
- [ECLARE: Extreme Classification with Label Graph Correlations](https://github.com/Extreme-classification/ECLARE)
- [GalaXC: Graph Neural Networks with Labelwise Attention for Extreme Classification](https://github.com/Extreme-classification/GalaXC)

## References

---
[1] K. Dahiya, N. Gupta, D. Saini, A. Soni, Y. Wang, K. Dave, J. Jiao, K. Gururaj, P. Dey, A. Singh, D. Hada, V. Jain, B. Paliwal, A. Mittal, S. Mehta, R. Ramjee, S. Agarwal, P. Kar and M. Varma. NGAME: Negative mining-aware mini-batching for extreme classification. In WSDM, Singapore, March 2023.

[2] K. Dahiya, A. Agarwal, D. Saini, K. Gururaj, J. Jiao, A. Singh, S. Agarwal, P. Kar and M. Varma. SiameseXML: Siamese networks meet extreme classifiers with 100M labels. In ICML, July 2021

[3] K. Dahiya, D. Saini, A. Mittal, A. Shaw, K. Dave, A. Soni, H. Jain, S. Agarwal, and M. Varma. Deepxml:  A deep extreme multi-label learning framework applied to short text documents. In WSDM, 2021.

[4] pyxclib: <https://github.com/kunaldahiya/pyxclib>

[5] The Extreme Classification Repository: <http://manikvarma.org/downloads/XC/XMLRepository.html>
