import torch.nn as nn

import torch

from torch.autograd import Variable


class WordGRU(nn.Module):
    def __init__(
        self,
        embedding_dim,
        vocab_size,
        batch_size=1,
        hidden_size=100,
        bidirectional=False,
        cuda=True
    ):

        super().__init__()

        self.batch_size = batch_size
        self.cuda = cuda
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim, hidden_size, bidirectional=bidirectional, batch_first=True
        )

    def _init_hidden(self):
        """
        Initialize the initial hidden and cell state
        :return:
        """
        if self.bidirectional:
            directions = 2
        else:
            directions = 1

        hidden = Variable(
            torch.zeros(
                directions, self.batch_size, self.hidden_size
            )
        )

        if self.cuda:
            return hidden.cuda()

        return hidden

    def seq_to_embedding(self, seq):
        """

        :param seq: A padded sequence of word indices
        :return:
        """
        embeds = []

        for s in seq:
            embeds.append(self.embedding(s))

        return torch.stack(embeds, dim=0)

    def forward(self, input, hidden):
        """

        :param hidden: Previous hidden state
        :param input: A padded sequence of word indices
        :return:
        """
        batch = self.seq_to_embedding(input)
        output, hidden = self.gru(batch, hidden)

        return output, hidden


class WordAttention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.hidden_size = hidden_size

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = torch.tanh

        self.word_context = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, word_outputs):

        o = self.linear(word_outputs)
        o = self.activation(o)
        o = torch.matmul(o, self.word_context)
        o = torch.mul(o, word_outputs)
        o = torch.sum(o, dim=1) # Sum along the words

        return o


class SentenceGRU(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=True):
        """

        :param input_size:
        :param hidden_size:
        :param bidirectional:
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size, hidden_size, bidirectional=bidirectional, batch_first=True
        )

    def forward(self, sentence_outputs, hidden):
        """

        :param sentence_outputs:
        :return:
        """
        return self.gru( sentence_outputs, hidden)


class SentenceAttention(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.linear = nn.Linear(input_size, input_size)
        self.activation = torch.tanh
        self.sentence_context = nn.Parameter(torch.randn(input_size,1))

    def forward(self, sent_outputs):

        o = self.linear(sent_outputs)
        o = self.activation(o)
        o = torch.matmul(o, self.sentence_context)
        o = torch.mul(o, sent_outputs)
        o = torch.sum(o, dim=1)

        return o