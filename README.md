# Poetry generator

A LSTM Network that learns patterns from a corpus of [Italian poetry](https://github.com/linhd-postdata/biblioteca_italiana) and generates poems.

# Folder structure

- `train_lstm.py` : the main entry point, it trains the NN. The `args` Namespace contains the customizable hyperparameters.
- `evaluation.py` : once the model has been trained and saved, this script uses the learnt embeddings and weights to generate a poem of fixed length.
- `utils/`:
  - `loadData.py` : preprocessing script
  - `dataset.py` : build dataset, extract inputs (n words) and labels (n+1 word)
  - `vectorizer.py` : Make tensors out of lines and n-grams
  - `lstm_module.py` : the NN class
  - `predictions.py` : functions used to generate, print and save poems
  - `utils.py` : function `print_info`, prints detailed infos about the layers and parameter numbers in the NN
- `data/` : contains the raw data and some predicted poems. (<small>The trained model was too large for Github</small>)

## Raw data structure

The format of each entry is as follows:

```json
{
    "url": "https://github.com/linhd-postdata/biblioteca_italiana/blob/master/xml/bibit000213",
    "author": "Dante Alighieri",
    "collection": "Il Fiore",
    "title": "I",
    "manually_checked": false,
    "text": [
        [
            {
                "verse": "Lo Dio d'Amor con su' arco mi trasse",
                "words": [
                    "Lo",
                    "Dio",
                    "d'Amor",
                    "con",
                    "su'",
                    "arco",
                    "mi",
                    "trasse"
                ]
            },
            ...
        ],
        ...
    ]
},
...
```
