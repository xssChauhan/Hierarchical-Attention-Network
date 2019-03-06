# Hierarchical-Attention-Network

PyTorch implementation of [Hierarchical Attention Network](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).


## Modules

### WordGRU: `models.WordGRU` 

A GRU layer that takes in batches of words and encodes them

### WordAttn: `models.WordAttn`

Attention Layer that takes in encoded words and generates Sentence Vectors

### SentenceGRU: `models.SentenceGRU`

GRU layer that accepts documents as sentences and encodes the sentences

### SentenceAttn: `models.SentenceAttn`

Attention Layer that takes in encoded sentences and generate document vectors



## Download and Prepare Data

Download the data from https://www.kaggle.com/utathya/imdb-review-dataset

Prepare the data for network using: `python imdb_data.py imdb_master.csv`

## Training the network

`python train.py ` 