import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def predict(model, words, word2index, index2word, out_file, top_pred=4, n_iterations=1000, bs=1, device="cpu"):
    model.eval()

    hidden, cell = model.set_hidden_states(bs=bs)

    # First parse what you already have
    for word in words:
        index = torch.tensor([[word2index[word]]])
        index = F.pad(index, pad=(
            model.seq_length - index.shape[1], 0), mode="constant", value=word2index["<PAD>"])

        index, hidden, cell = index.to(
            device), hidden.to(device), cell.to(device)

        output, (hidden, cell) = model(index, (hidden, cell))

        hidden, cell = hidden.detach(), cell.detach()

    # Make a prob distribution out of it
    output = torch.softmax(output, dim=1)
    _, top_indices = torch.topk(output[0], k=top_pred)
    choices = top_indices.tolist()

    if word2index[word] in choices:
        choices.remove(word2index[word])
    choice = np.random.choice(choices)

    words.append(index2word[choice])

    for _ in tqdm(range(n_iterations), desc="Rubinettando versi..."):
        index = torch.tensor([[choice]]).long()
        index = F.pad(index, pad=(
            model.seq_length - index.shape[1], 0), mode="constant", value=word2index["<PAD>"])

        index, hidden, cell = index.to(
            device), hidden.to(device), cell.to(device)

        output, (hidden, cell) = model(index, (hidden, cell))

        hidden, cell = hidden.detach(), cell.detach()

        output = torch.softmax(output, dim=1)

        _, top_indices = torch.topk(output[0], k=top_pred)
        choices = top_indices.tolist()

        if choice in choices:
            choices.remove(choice)
        choice = np.random.choice(choices)
        words.append(index2word[choice])
    with open(out_file, "a", encoding="utf-8") as pred:
        print("\n")
        for i in range(0, len(words), 5):
            verse = words[i:i+5]
            verse = " ".join(verse)
            print(verse)
            pred.write(verse + "\n")
        pred.write("\n" + "#"*40 + "\n")
