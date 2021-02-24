import torch
import torch.nn as nn
from torch.autograd import Variable


class PoetryGenerator(nn.Module):
    def __init__(self, vocab_size, embs, hidden, seq_len, bs, layers=2, device='cpu', debug=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embs_size = embs
        self.hidden_size = hidden
        self.seq_length = seq_len - 1  # since the last item must be predicted
        self.num_layers = layers
        self.bs = bs

        self.device = device
        self.debug = debug

        self.embeddings = nn.Embedding(self.vocab_size, self.embs_size)
        self.lstm = nn.LSTM(self.embs_size, self.hidden_size,
                            num_layers=self.num_layers, bidirectional=True, dropout=0.2)

        self.dense = nn.Linear(2 * self.hidden_size *
                               self.seq_length, self.vocab_size)

    def init_embeddings(self):
        torch.nn.init.uniform(self.embeddings.weight, -1, 1)

    def set_hidden_states(self, bs=None):
        if bs is None:
            bs = self.bs
        hidden = Variable(torch.zeros(self.num_layers * 2,
                                      bs, self.hidden_size), requires_grad=True)
        cell = Variable(torch.zeros(self.num_layers * 2, bs,
                                    self.hidden_size), requires_grad=True)
        return (hidden, cell)

    def print_info(self, desc, x):
        if self.debug:
            print(f"{desc} : {x.shape}, {x.device}, {x.requires_grad}")

    def forward(self, x, previous_state):

        self.print_info("Input", x)
        self.print_info("Hidden", previous_state[0])
        self.print_info("Cell state", previous_state[1])

        x = self.embeddings(x)
        self.print_info("Embeddings", x)

        # RESHAPING
        #x = x.view(self.seq_length, self.bs, -1)
        if self.training:
            x = x.view(-1, self.bs, self.embs_size)
            self.print_info("Reshaped for LSTM", x)
        else:
            x = x.view(-1, 1, self.embs_size)  # seq_len, 1, embs_size

        # LSTM
        x, (hidden, cell) = self.lstm(x, previous_state)
        self.print_info("Out from LSTM", x)
        self.print_info("Hidden from LSTM", hidden)
        self.print_info("Cell from LSTM", cell)

        # RESHAPE FOR LINEAR
        if self.training:
            x = x.view(self.bs, -1)
            self.print_info("Reshaped for Linear", x)
        else:
            x = x.view(1, -1)

        x = self.dense(x)
        self.print_info("After Dense layer", x)

        return x, (hidden, cell)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
