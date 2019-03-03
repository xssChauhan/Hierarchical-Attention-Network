import sys

from typing import List

import itertools

import pandas as pd
import nltk
import contractions
import joblib

import dask.dataframe as dd

import numpy as np

def load_csv(filename):
    df = pd.read_csv(filename, encoding="latin-1")

    return df


def clean_text(string):

    string = string.lower()
    string = contractions.fix(string)

    return string


def sentences_word_tokenize(sentences: List[str]):
    output = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [
            e for e in words if e.isalnum()
        ]
        if len(words):
            output.append(words)

    return output


def process_and_persist(df):

    # Clean
    df.review = df.review.map(clean_text)

    #Convert to sentences
    df.sentences = df.review.map(
            nltk.sent_tokenize
    )

    # Break down sentences into words
    df.sentence_words = df.sentences.map(
        sentences_word_tokenize
    )

    # Calcualte Sentence lengths
    df.sentence_len = df.sentences_words.map(len)

    # Calculate Word Counts
    df.word_count = df.sentence_words.map(
        lambda sentences: sum(len(e) for e in sentences)
    )

    df.max_word_count = df.sentence_words.map(
        lambda sentences: max((len(e) for e in sentences))
    )

    #Modify label
    df.binary_label = df.label.map(lambda label: 1 if label == "pos" else 0)

    return df


def process_df(df, outfile):
    df["sentences"] = np.nan
    df["sentence_len"] = np.nan
    df["sentence_words"] = np.nan
    df["word_count"] = np.nan
    df["binary_label"] = np.nan
    df["max_word_count"] = np.nan


    ddf = dd.from_pandas(df, npartitions=3)

    res = ddf.map_partitions(process_and_persist, meta = df)

    res = res.compute()
    res = res.sort_values("sentence_len", ascending=False)
    joblib.dump(res, outfile)

    return res


def split_test_train(df):
    train = df[df.type == "train"]
    test = df[df.type == "test"]

    train = process_df(train, "train.pkl")
    test =process_df(test, "test.pkl")


if __name__ == "__main__":
    filename = sys.argv[1]

    split_test_train((load_csv(filename)))