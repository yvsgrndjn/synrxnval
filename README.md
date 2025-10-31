# synrxnval

**synrxnval** is a Python package to validate *synthetic chemical reactions* (that can be generated with the **synrxnforge** package) by exploiting chemical knowledge of 2 seq2seq Transformer models.

For a given synthetic reaction (`A>>C`), the reagents `B` are first predicted. Then, the combination of tagged reactants and reagents (`A!>B`) are submitted to a forward reaction prediction model*, that predicts the outcome `C'` of the given mixture.

If `C'== C` and the confidence score of the prediction is >0.95, the synthetic reaction is considered valid and is kept. 

---

## Installation

You can install `synrxnval` directly from source:

```bash
conda create -n synrxnval python=3.10 -y
conda activate synrxnval
conda install -c conda-forge rdkit -y

git clone https://github.com/yvsgrndjn/synrxnval.git
cd synrxnval
pip install -e .
```

---


## Setup

The input file can have several formats, the easiest is the output parquet file from **synrxnforge**.

you need first to download the reagents prediction (T2) model and the forward tag prediction model (T3*). Models trained on USPTO data are available 
[https://zenodo.org/records/14017743?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImY2NmZmNTkxLTE2YzYtNDU0OS04NjAzLTJiNzg1YzFhMGQ5NSIsImRhdGEiOnt9LCJyYW5kb20iOiJmMjBhNjlkNDRkYTE1YWMwNDVjODQ2YjkwOTQ1ZjgyNCJ9.DRQXQBjRcv6MTW1hEYDSYZ6j11dmKBAQI-nytHBTHKu66KYTS3TgriJW_pOTfayHcditLS4MNKa9okI4FLSD2Q|here]. 


## Run validation

```bash
$ synrxnval python -m --input path/to/parquet
                      --t2_model path/to/model
                      --t3_model path/to/model
                      --out_dir path/to/out
```
