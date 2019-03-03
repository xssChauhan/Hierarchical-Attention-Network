from data_utils import *

from models import *

import torch.nn as nn
import torch.optim as optim

import torch
import tqdm


def train(filename, learning_rate=0.01, epochs=100, cuda=True):
    """

    :param learning_rate:
    :param epochs:
    :param filename:
    :return:
    """

    glove, vocab2index, index2vocab = load_glove("glove/glove.6B.100d.txt", 100)

    glove = torch.from_numpy(np.array(
        list(glove.values())
    ))

    network = HAN(
        vocab_size=len(vocab2index),
        embedding_dim=100,
        word_hidden_size=50,
        sent_hidden_size=50,
        num_labels=2,
        bidirectional=True,
        cuda=cuda,
        embedding=glove
    )

    loss = nn.NLLLoss()
    optimizer = optim.SGD(
        network.parameters(),
        lr=learning_rate,
        momentum=0.9
    )
    if cuda:
        loss.cuda()

    for epoch in tqdm.tqdm(range(epochs), desc="Epoch"):
        data_generator = generate_data(filename, vocab2index)

        for batch, labels in data_generator:
            if cuda:
                labels = labels.long().cuda()
            op = network(batch)
            l = loss(op, labels)
            l.backward()
            optimizer.step()

if __name__ == "__main__":
    train("train.pkl", cuda=True)