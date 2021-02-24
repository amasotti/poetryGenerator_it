import torch
from torch.utils.data import Dataset
import json


class Data(Dataset):
    def __init__(self, padded_corpus_path, vocab):

        self.padded_corpus_path = padded_corpus_path
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.padded_corpus = self.load_corpus()
        self.maxLength = self.padded_corpus.shape[1]

        self.inputs, self.labels = self.make_input_labels()

    def load_corpus(self):
        return torch.load(self.padded_corpus_path)

    def __len__(self):
        return self.padded_corpus.shape[0]

    def make_input_labels(self):
        # For each chunk take the firs but last words as input
        inputs = self.padded_corpus[:, :-1]
        # Take the last word as label
        labels = self.padded_corpus[:, -1]
        return inputs, labels

    def __getitem__(self, i):
        X = self.inputs[i]
        y = self.labels[i].long()
        return X, y
