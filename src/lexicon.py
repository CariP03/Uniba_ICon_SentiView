import pandas as pd
from nltk.tokenize import word_tokenize

from data_load import read_dataset

# read dataset
TRAIN_PATH = "..\\data\\aclImdb\\train"
df_train = read_dataset(TRAIN_PATH)

vocab = set()
for text in df_train["text"]:
    tokens = word_tokenize(text.lower())

    # remove punctuation
    for token in tokens:
        if any(c.isalpha() for c in token):
            vocab.add(token)