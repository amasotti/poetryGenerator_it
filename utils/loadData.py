import glob
import json
import numpy as np
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
import re

REPOSITORY = "data/biblioteca_italiana/json"
tokenizer = WordPunctTokenizer()


def findFiles(path):
    return glob.glob(path + "/*.json")


def save_corpus(corpus, path, npPath=None, np_save=False):
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(corpus))
    if np_save:
        np.save(npPath, corpus)


def buildCorpus():
    fileList = findFiles(REPOSITORY)
    corpus = []
    for author in tqdm(fileList, "Looping over Authors"):
        with open(author, "r", encoding="utf-8") as fp:
            collection = json.load(fp)
            for work in collection:
                work_text = work['text']
                if work_text != []:
                    for verses in work_text[0]:
                        corpus.append(verses['verse'])
                else:
                    for verses in work_text:
                        corpus.append(verses['verse'])
    save_corpus(corpus, "data/corpus.txt", "data/corpus.npy", np_save=True)
    return corpus


def tokenize(corpus):
    tokenized_corpus = []
    for sent in corpus:
        new_sent = re.sub(r"\W+", " ", sent, 0, re.MULTILINE)
        new_sent = re.sub(r"\d+", "", new_sent, 0, re.MULTILINE)
        new_sent = re.sub(r"\s+", " ", new_sent, 0, re.MULTILINE)
        new_sent = tokenizer.tokenize(new_sent)
        # TODO: Stopwords?
        tokenized_corpus.append(new_sent)
    np.save("data/tokenized_corpus.npy", tokenized_corpus)
    with open("data/cleaned_corpus.txt", "w", encoding="utf-8") as fp:
        for sent in tokenized_corpus:
            fp.write(" ".join(sent) + "\n")
    return tokenized_corpus


# Extract texts from file
corpus = buildCorpus()
# Tokenize
corpus = tokenize(corpus)
