"""
Take in (text, labels)

1. Break down documents into sentences
2. Break down sentences into words
3. Data : [n_docs, sentences, words]
4. Since the HAN expects sentences the data needs to be packed as:
    [docs_in_batch, max_sents, max_words]


For WordGRU, batch dimension is for the sentences.

For SentenceGRU, batch dimension is for the documents.


Strategy for Data:
    1. Take a batch of docs: batch_docs
    2. Each document will be processed at a time by the word encoder as batch of sentences
    3. The sentence encoder processes a batch of documents at once
    4. So, to get sentence representations of the batch, each document in the batch has to be
    iterated, and the outputs from word encoder to be stacked to feed the sentence encoder.
"""

import numpy as np
import joblib

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence


def load_glove(glove_file, dimension: int):
    print("Loading Glove Model")
    f = open(glove_file, 'r')
    model = {}
    vector_size = dimension
    for line in f:
        split_line = line.strip().split()
        word = " ".join(split_line[0:len(split_line) - vector_size])
        embedding = np.array([float(val) for val in split_line[-vector_size:]])
        model[word] = embedding

    model["<pad>"] = np.zeros(dimension)
    print("Done.\n" + str(len(model)) + " words loaded!")


    vocab2index = {}
    index2vocab = {}

    for index, word in enumerate(model.keys()):
        vocab2index[word] = index
        index2vocab[index] = word

    return model, vocab2index, index2vocab


def load_data(filename):
    """
    Load all the data
    :return:
    """
    df = joblib.load(filename)

    usable_df = df[
        [
            "sentence_words",
            "sentence_len",
            "binary_label"

        ]
    ]

    return usable_df


def convert_to_indices(sentences, vocab2index, cuda=False):
    output = []
    lengths = []
    for sentence in sentences:
        indexed = [vocab2index.get(word, vocab2index.get('<unk>')) for word in sentence]
        lengths.append(len(indexed))
        tensor = torch.LongTensor(indexed)
        output.append(tensor)

    sorted_pairs = sorted(
        zip(output, lengths),
        key= lambda data: data[1],
        reverse=True
    )
    lengths = [l for _,l in sorted_pairs]
    # import ipdb; ipdb.set_trace()

    seq = pad_sequence([s for s,_ in sorted_pairs], batch_first=True, padding_value=len(vocab2index)-1)

    return seq


def prepare_mini_batch(mini_batch, vocab2index, cuda=False):
    """
    Prepare mini batch for traning by padding to appropriate lengths and batch sizes

    :param mini_batch:
    :return:

    """
    indexed = []
    for sentence_words in mini_batch.sentence_words:
        # sentence_words is list of sentences in the document
        # We convert each document to sequence at a time
        indices = convert_to_indices(sentence_words, vocab2index, cuda=cuda)
        # indices = torch.LongTensor(indices)
        indexed.append(indices)

    # padded = pad_sequence(indexed)
    return indexed, torch.LongTensor(mini_batch.binary_label.tolist())


def _get_batch_ranges(length, batch_size):
    """

    :param length:
    :param batch_size:
    :return:
    """
    return range(0, length, batch_size)


def generate_data(filename, vocab2index, batch_size=5, cuda=False):
    df = load_data(filename)

    batches = range(0,len(df), batch_size)
    start = 0
    for end in batches[1:]:
        yield prepare_mini_batch(df[start:end], vocab2index, cuda=cuda)
        start = end
