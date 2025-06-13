import string
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer as wnl
import contractions

from config import INTENSIFIERS, NEGATORS

def tokenize_dataframe(df):
    vocab = set()
    for text in df["text"]:
        tokens = clean_tokenize(text)
        for token in tokens:
            vocab.add(token)

    return vocab


def clean_tokenize(text: str):
    def get_wordnet_pos(treebank_tag):
        # convert NLTK's tags to WordNet's tags
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN

    # source: https://stackoverflow.com/questions/17245123/getting-adjective-from-an-adverb-in-nltk-or-other-nlp-library
    # author: alvas (accessed 13 June 2025)
    def convert_adv_to_adj(adverb):
        for ss in wn.synsets(adverb, pos=wn.ADV):
            for lemmas in ss.lemmas():  # all possible lemmas
                pert = lemmas.pertainyms()  # all possible pertainyms
                if pert:
                    return pert[0].name()

        return None

    tokens = word_tokenize(contractions.fix(text).lower())
    pos_tags = pos_tag(tokens)
    lemmatizer = wnl()

    clean_tokens = []
    for token, tag in pos_tags:
        token = token.strip(string.punctuation)

        if any(c.isalpha() for c in token):  # remove invalid tokens
            if token in NEGATORS:
                clean_tokens.append(token)
                continue
            if token in INTENSIFIERS:
                clean_tokens.append(token)
                continue

            pos = get_wordnet_pos(tag)
            if pos:
                lemma = lemmatizer.lemmatize(token, pos)
            else:
                # if get_wordnet_pos returns None
                lemma = lemmatizer.lemmatize(token)

            # try to convert adverb to adjective
            if pos == wn.ADV:
                adj = convert_adv_to_adj(token)
                if adj:
                    lemma = adj

            clean_tokens.append(lemma)

    return clean_tokens