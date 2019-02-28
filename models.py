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
        bidirectional=True,
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
        """

        :param hidden_size: Number of features in the incoming word vecs
        """

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

        :param input_size: Size of incoming sentence vecs
        :param hidden_size: Hidden size of GRU unit
        :param bidirectional: If the GRU should be bidirectional
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

        :param sentence_outputs: Sentence vecs from the word attention layer
        :return:
        """
        return self.gru(sentence_outputs, hidden)


class SentenceAttention(nn.Module):

    def __init__(self, input_size):
        """
        :param input_size: Size of vectors from the sentence GRU
        """

        super().__init__()

        self.linear = nn.Linear(input_size, input_size)
        self.activation = torch.tanh
        self.sentence_context = nn.Parameter(torch.randn(input_size,1))

    def forward(self, sent_outputs):
        """

        :param sent_outputs: Sentence vectors from sentence GRU
        :return:
        """

        o = self.linear(sent_outputs)
        o = self.activation(o)
        o = torch.matmul(o, self.sentence_context)
        o = torch.mul(o, sent_outputs)
        o = torch.sum(o, dim=1)

        return o


class OutputLayer(nn.Module):

    def __init__(self, input_size, num_labels):
        """

        :param input_size: Number of features in the incoming vector
        :param num_labels: Number of labels in the data
        """

        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels

        self.linear = nn.Linear(input_size, num_labels)
        self.softmax = nn.LogSoftmax()

    def forward(self, doc_vector):
        """

        :param doc_vector: Document Vector
        :return:
        """

        o = self.linear(doc_vector)
        o = self.softmax(o)

        return o


class HAN(nn.Module):

    def __init__(self, vocab_size, embedding_dim ,word_hidden_size, sent_hidden_size, num_labels, batch_size, bidirectional, cuda):
        """

        :param vocab_size:
        :param word_hidden_size:
        :param sent_hidden_size:
        :param num_labels:
        :param batch_size:
        :param bidirectional:
        :param cuda:
        """

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.cuda = cuda

        self.directions = 2 if bidirectional else 1

        self.word_gru = WordGRU(
            embedding_dim, vocab_size, word_hidden_size, bidirectional, cuda
        )
        self.word_attn = WordAttention(word_hidden_size * directions)

        self.sentence_gru = SentenceGRU(
            word_hidden_size * directions, sent_hidden_size, bidirectional
        )
        self.sentence_attn = SentenceAttention(
            directions * sent_hidden_size
        )

        self.output_layer = OutputLayer(
            directions * sent_hidden_size
        )


        def forward(self, inputs):
            """

            :param self:
            :param inputs:
            :return:
            """
            # Get encoded sentences
            # Unsqueeze them
            # Combine them
            return

        def set_embedding(self, embedding):
            """
            Initialize the Embedding

            :param self:
            :param embedding:
            :return:
            """
            return