import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
from tqdm import tqdm


class Vectorizer(object):
    def __init__(self, corpus_path, outpath, inpath=None, vocab_path=None):
        self.corpus_path = corpus_path
        self.vocab_path = vocab_path
        self.outpath = outpath
        self.inpath = inpath

        self.vocab = dict()
        self.maxLength = 0

        if inpath is None:  # First processing
            # Create corpus & update dictionary
            self.corpus = self.load_corpus()
            # Pad corpus
            self.corpus = self.pad_corpus()
        else:
            # Load corpus
            self.corpus = self.load_padded()
            self.maxLength = self.corpus.shape[1]
            # Load vocab
            self.load_vocab()

    def load_corpus(self):
        # Load the raw data
        with open(self.corpus_path, "r", encoding="utf-8") as fp:
            corpus = fp.readlines()

        # Build the tokenized corpus
        corpus = self.tokenize(corpus)

        # Extract the dictionary
        self.vocab = self.build_vocab(corpus)
        # Save the dictionary

        self.save_vocab()

        return corpus

    @staticmethod
    def tokenize(corpus):
        try:
            from nltk.tokenize import WordPunctTokenizer
            tokenizer = WordPunctTokenizer()
        except:
            raise NotImplementedError

        tokenized = []
        for sent in tqdm(corpus, "Tokenisation:"):
            new_sent = sent.lower()
            new_sent = tokenizer.tokenize(sent)
            new_sent = [w for w in new_sent if w not in ["...", "'"]]
            tokenized.append(new_sent)

        return tokenized

    @staticmethod
    def build_vocab(tokenized_corpus):
        vocab = dict()
        for sent in tokenized_corpus:
            for w in sent:
                if w not in vocab:
                    vocab[w] = len(vocab)
        vocab["<PAD>"] = len(vocab)
        return vocab

    def vectorize(self):
        """
        Transforms each sentence in the correspoding list of indices

        """
        vectorized_corpus = []
        for sent in tqdm(self.corpus, "Vectorization"):
            # Retrieve indices
            indices = [self.vocab[w] for w in sent]
            vectorized_corpus.append(indices)
        return vectorized_corpus

    def make_ngrams(self):
        """ split sentences into growing ngrams """
        vectorized_corpus = self.vectorize()
        ngrams_corpus = []
        for vector in vectorized_corpus:
            for i in range(1, len(vector)-1):
                ngram = vector[:i+1]
                ngrams_corpus.append(ngram)
        # Test to see how the chunkes look like
        """with open("data/ngrams_corpus.txt", "w", encoding="utf-8") as fp:
            for line in ngrams_corpus:
                fp.write(str(line) + "\n")"""
        return ngrams_corpus

    def pad_corpus(self):
        pad_value = self.vocab["<PAD>"]
        corpus = self.make_ngrams()
        max_length = max([len(x) for x in self.corpus])
        padded_corpus = torch.empty((len(corpus), max_length))

        for i, chunk in tqdm(enumerate(corpus), "Pad corpus", total=len(corpus)):
            tensor = torch.tensor(chunk).long()
            tensor = F.pad(tensor, pad=(max_length - len(chunk),
                                        0), mode="constant", value=pad_value)
            padded_corpus[i] = tensor
        # Values should be integer
        padded_corpus = padded_corpus.long()
        self.maxLength = max_length
        return padded_corpus

    def save_corpus(self):
        torch.save(self.corpus, self.outpath)

    def load_vocab(self):
        with open(self.vocab_path, "r", encoding="utf-8") as fp:
            self.vocab = json.loads(fp.read())

    def save_vocab(self):
        with open(self.vocab_path, "w", encoding="utf-8") as fp:
            json.dump(self.vocab, fp, ensure_ascii=False)

    def load_padded(self):
        return torch.load(self.inpath)

    @property
    def max_length(self):
        return self.maxLength


if __name__ == '__main__':
    vec = Vectorizer(corpus_path="data/cleaned_corpus.txt",
                     outpath="data/padded_corpus.pt", inpath=None,
                     vocab_path="data/word2index.json")
    vec.save_corpus()
    print(f"Max_length: {vec.max_length}")
