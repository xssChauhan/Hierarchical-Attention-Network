from data_utils import *

from models import *

import torch.nn as nn
import torch.optim as optim

import torch
import tqdm


def train(filename, learning_rate=0.01, epochs=100, cuda=True):
    """

    :param cuda:
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

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        network.parameters(),
        lr=learning_rate,
        momentum=0.9
    )
    if cuda:
        loss.cuda()

    for epoch in tqdm.tqdm(range(epochs), desc="Epoch"):
        training_loss = 0.0
        validation_loss = 0.0

        data_generator = generate_data(filename, vocab2index, batch_size=64)
        network.train()
        for batch, labels in data_generator:
            if cuda:
                labels = labels.long().cuda()
            optimizer.zero_grad()
            op = network(batch)
            # print(op)
            l = loss(op, labels)
            # print(l)
            # print(l.item())
            l.backward()
            training_loss += l.item()*64
            optimizer.step()

        # network.eval()
        #
        # test_data_generator = generate_data("test.pkl", vocab2index, batch_size=32)
        # for batch, labels in test_data_generator:
        #     if cuda:
        #         labels = labels.long().cuda()
        #     output = network(batch)
        #     val_l = loss(output, labels)
        #     validation_loss += val_l*32

        # validation_loss /= 25000

        training_loss /= 50000

        with open("output.txt", "w") as f:
            f.write("Epoch: {}\t Training Loss: {:.6f} \t Validation Loss: -".format(
                epoch, training_loss
            ))


if __name__ == "__main__":
    train("train.pkl", cuda=True, learning_rate=0.001)