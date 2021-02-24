import json
import torch

from utils.lstm_module import PoetryGenerator
from utils.predictions import predict

################################################################
START = ["amore", "ha"]


################################################################
with open("data/word2index.json", "r", encoding="utf-8") as fp:
    vocab = json.loads(fp.read())

index2word = {i: w for w, i in vocab.items()}

netz = PoetryGenerator(vocab_size=len(vocab),
                       seq_len=15,
                       embs=200,
                       hidden=100,
                       bs=1,
                       device="cpu",
                       debug=False)

netz.load_state_dict(torch.load("data/models/model.pth"))

################################################################

predict(model=netz,
        words=START,
        word2index=vocab,
        index2word=index2word,
        out_file="data/models/evaluation.txt",
        top_pred=9,
        n_iterations=40)
