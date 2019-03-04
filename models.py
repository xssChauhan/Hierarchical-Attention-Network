import torch.nn as nn

import torch

from torch.autograd import Variable

from torch.nn.utils.rnn import pad_sequence


class WordGRU(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_size=100, bidirectional=True, embedding=None, is_cuda=False):

        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.is_cuda = is_cuda

        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        else:
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

        hidden = Variable(torch.zeros(directions, self.batch_size, self.hidden_size))

        if self.cuda:
            return hidden.cuda()

        return hidden

    def seq_to_embedding(self, seq):
        """

        :param seq: A padded sequence of word indices
        :return:
        """
        embeds = []

        if self.is_cuda:
            seq = seq.cuda()

        for s in seq:
            embeds.append(self.embedding(s))

        return torch.stack(embeds, dim=0)

    def forward(self, input):
        """

        :param hidden: Previous hidden state
        :param input: A padded sequence of word indices
        :return:
        """
        batch = self.seq_to_embedding(input)
        # print("Dimension of input to WordGRU ", input.shape)
        output, _ = self.gru(batch.float())
        # print("Dimension of output from WordGRU ", output.shape)
        return output


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
        # print("Dimension of input to WordAttn", word_outputs.shape)
        o = self.linear(word_outputs)
        # print("Dimension of input to WordAttn", word_outputs.shape)
        o = self.activation(o)
        o = torch.matmul(o, self.word_context)
        o = torch.mul(o, word_outputs)
        o = torch.sum(o, dim=1)  # Sum along the words

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

    def forward(self, sentence_outputs):
        """

        :param sentence_outputs: Sentence vecs from the word attention layer
        :return:
        """
        output, _ = self.gru(sentence_outputs)
        return output


class SentenceAttention(nn.Module):
    def __init__(self, input_size):
        """
        :param input_size: Size of vectors from the sentence GRU
        """

        super().__init__()

        self.linear = nn.Linear(input_size, input_size)
        self.activation = torch.tanh
        self.sentence_context = nn.Parameter(torch.randn(input_size, 1))

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
        self.softmax = nn.Softmax()

    def forward(self, doc_vector):
        """

        :param doc_vector: Document Vector
        :return:
        """

        o = self.linear(doc_vector)
        o = self.softmax(o)

        return o


class HAN(nn.Module):
    # TODO Take in a batch of documents. Iterate over each document, and pass it through WordGRU. Accumulate results for all documents from WordGRU and WordAttn
    # TODO Pass the accumulated results from Word Encoder to Sentence Encoder to Output Layer

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        word_hidden_size,
        sent_hidden_size,
        num_labels,
        bidirectional,
        cuda,
        embedding=None
    ):
        """

        :param vocab_size:
        :param word_hidden
        self.embedding = embedding_size:
        :param sent_hidden_size:
        :param num_labels:
        :param bidirectional:
        :param cuda:
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.num_labels = num_labels
        self.bidirectional = bidirectional
        self.cuda = cuda
        self.embedding = embedding

        self.directions = 2 if bidirectional else 1

        self.word_gru = WordGRU(
            embedding_dim, vocab_size, word_hidden_size, bidirectional, embedding, is_cuda=self.cuda
        )
        self.word_attn = WordAttention(word_hidden_size * self.directions)

        self.sentence_gru = SentenceGRU(
            word_hidden_size * self.directions, sent_hidden_size, bidirectional
        )
        self.sentence_attn = SentenceAttention(self.directions * sent_hidden_size)

        self.output_layer = OutputLayer(
            self.directions * sent_hidden_size, self.num_labels
        )

        if self.cuda:
            self.word_gru.cuda()
            self.word_attn.cuda()
            self.sentence_gru.cuda()
            self.sentence_attn.cuda()
            self.output_layer.cuda()

    def forward(self, documents):
        """
        # TODO should this function be responsible for padding the sequence as well? I think no

        :param self:
        :param documents: A 3D array of shape [batch_size, sents, words]
        :return:
        """
        # Get encoded sentences
        # Unsqueeze them
        # Combine them

        # Encode Words using WordGRU and WordAttn

        sentence_vectors = []
        for document in documents:
            encoded_words = self.word_gru(document)
            encoded_sentence = self.word_attn(encoded_words)
            sentence_vectors.append(encoded_sentence)

        document_tensor = pad_sequence(sentence_vectors, batch_first=True)
        # print("Size of Doc Vector ", document_tensor.size())

        doc_vec = self.sentence_attn(self.sentence_gru(document_tensor))

        output = self.output_layer(doc_vec)

        return output
