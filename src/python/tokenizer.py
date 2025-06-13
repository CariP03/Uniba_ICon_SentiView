from nltk.tokenize import word_tokenize


def tokenize_dataframe(df):
    vocab = set()
    for text in df["text"]:
        tokens = word_tokenize(text.lower())

        # remove punctuation
        for token in tokens:
            if any(c.isalpha() for c in token):
                vocab.add(token)

    return vocab
