from data_utils import *

from models import *

import torch.nn as nn
import torch.optim as optim

import torch
import tqdm

from utils import plot_grad_flow


def train(filename, learning_rate=0.01, epochs=100, cuda=True, batch_size=32):
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
    optimizer = optim.Adam(
        network.parameters(),
        lr=learning_rate
    )
    if cuda:
        loss.cuda()

    for epoch in tqdm.tqdm(range(epochs), desc="Epoch"):
        training_loss = 0.0
        validation_loss = 0.0

        data_generator = generate_data(filename, vocab2index, batch_size=batch_size, cuda=cuda)
        network.train()
        for batch, labels in data_generator:

            if cuda:
                labels = labels.cuda()
            optimizer.zero_grad()
            op = network(batch)
            # print(op)
            l = loss(op, labels)
            # print(l)
            # print(l.item())
            l.backward()
            training_loss += l.item()*batch_size
            optimizer.step()
            nn.utils.clip_grad_value_(network.parameters(), 0.25)
            plot_grad_flow(network.named_parameters())

        with torch.no_grad():
            network.eval()

            test_data_generator = generate_data("test.pkl", vocab2index, batch_size=batch_size, cuda=cuda)
            for batch, labels in test_data_generator:
                if cuda:
                    labels = labels.long().cuda()

                output = network(batch)
                val_l = loss(output, labels)
                validation_loss += val_l*batch_size

            validation_loss /= 25000

        training_loss /= 50000

        with open("output_1.txt", "a") as f:
            f.write("Epoch: {}\t Training Loss: {:.6f} \t Validation Loss: {:.6f}\n".format(
                epoch, training_loss, validation_loss
            ))


if __name__ == "__main__":
    train("train_s.pkl", cuda=False, learning_rate=0.005, batch_size=32, epochs=10)