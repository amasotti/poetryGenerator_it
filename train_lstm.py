from argparse import Namespace
from utils.predictions import predict
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.optim as O
from torch.utils.data import DataLoader
from utils.utils import print_infos
from utils.dataset import Data
from utils.lstm_module import PoetryGenerator

import matplotlib.pyplot as plt

args = Namespace(
    train_file='data/padded_corpus.pt',
    vocab="data/word2index.json",
    load_model=True,
    debug=False,
    lr=1e-3,
    lr_decay=0.90,
    epochs=100,
    seq_size=None,
    batch_size=80,
    embedding_size=200,
    hidden_size=100,
    words=["come", "i", "matti", "a", "mezzogiorno"],
    save_each=500,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    # device="cpu",
    model="data/models/model.pth",
    predictions="data/models/predictions.txt"
)

# Load data, vocab, index2word
with open(args.vocab, "r", encoding="utf-8") as v:
    vocab = json.loads(v.read())

dataset = Data(padded_corpus_path=args.train_file,
               vocab=vocab)
args.seq_size = dataset.maxLength
index2word = {i: w for w, i in vocab.items()}


netz = PoetryGenerator(vocab_size=len(vocab),
                       seq_len=args.seq_size,
                       embs=args.embedding_size,
                       hidden=args.hidden_size,
                       bs=args.batch_size,
                       device=args.device,
                       debug=args.debug)

if args.load_model:
    netz.load_state_dict(torch.load(args.model))
    print("Model loaded")

if args.device == "cuda":
    netz.to(args.device)
    print("Using CUDA")


######### PRINT INFORMATION ABOUT THE MODEL #############
print(netz)
print_infos(model=netz)

dataloader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

#optimizer = O.Adam(netz.parameters(), lr=args.lr)
optimizer = O.RMSprop(netz.parameters(), lr=args.lr, momentum=.7)
criterion = nn.CrossEntropyLoss(reduction='mean')
running_loss = 0
all_losses = []


def train(netz=netz, optimizer=optimizer, criterion=criterion, running_loss=running_loss, all_losses=all_losses):
    epoch_bar = tqdm(total=args.epochs, desc="Epoch routine")
    num_batches = len(dataset) / (args.batch_size)

    train_bar = tqdm(total=num_batches, desc="Training Routine")

    for epoch in range(args.epochs):

        hidden, cell = netz.set_hidden_states()

        if args.device == "cuda":
            hidden, cell = hidden.cuda(), cell.cuda()

        for i, (x, y) in enumerate(dataloader):
            if args.device == "cuda":
                x = x.cuda()
                y = y.cuda()

            # Start training
            netz.train()

            # Reset gradients
            netz.zero_grad()

            # Make prediction
            out, (hidden, cell) = netz(x, (hidden, cell))

            hidden, cell = hidden.detach(), cell.detach()

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().item())

            if i > 0 and i % args.save_each == 0:
                train_bar.set_postfix(
                    epoch=epoch, loss=running_loss / args.save_each)
                all_losses.append(running_loss / args.save_each)
                running_loss = 0

                # make prediction
                predict(model=netz, words=args.words, word2index=vocab, index2word=index2word,
                        out_file=args.predictions, top_pred=5, n_iterations=5, device=args.device)
                #args.words = ['così', "inizia", 'il', 'poema']

                # Save the model if improved
                if len(all_losses) > 1:
                    if all_losses[-1] < all_losses[-2]:
                        netz.save_model(args.model)
                        print("\nLoss improved, Save model\n")
            train_bar.update()
        # Kind of scheduler
        if epoch > 0 and epoch % 5 == 0:
            args.lr *= args.lr_decay
            #optimizer = O.Adam(netz.parameters(), lr=args.lr)
            optimizer = O.RMSprop(netz.parameters(), lr=args.lr, momentum=.7)

        # Restart counting batches
        train_bar.n = 0
        epoch_bar.set_postfix(epoch=epoch, loss=all_losses[-1], lr=args.lr)
        epoch_bar.update()


#### START TRAINING ####
train()


##########################################
#  PLOT LOSSES
##########################################

plt.figure()
plt.plot(all_losses)
plt.show()
