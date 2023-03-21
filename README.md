# ngame

Code for _NGAME: Negative mining-aware mini-batching for extreme classification_

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

## Example use cases

---

### A single learner

The given code can be utilized as follows. A json file is used to specify architecture and other arguments. Please refer to the full documentation below for more details.

```bash
./run_main.sh 0 NGAME LF-AmazonTitles-131K 0 108
```

## Full Documentation

```txt
./run_main.sh <gpu_id> <type> <dataset> <version> <seed>
* gpu_id: Run the program on this GPU.
* type
  NGAME uses DeepXML[3] framework for training. The classifier is trained in M-IV.
  - NGAME: The intermediate representation is not fine-tuned while training the classifier (more scalable; suitable for large datasets).
  - NGAME++: The intermediate representation is fine-tuned while training the classifier (leads to better accuracy on some datasets). #TODO
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
- [ ] tokenizer

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