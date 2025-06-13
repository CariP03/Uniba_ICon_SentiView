from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize


def get_swn_score(word):
    synsets = wn.synsets(word)
    scores = []

    for syn in synsets:
        swn_syn = swn.senti_synset(syn.name())
        net_score = swn_syn.pos_score() - swn_syn.neg_score()
        scores.append(net_score)

    if scores:
        return sum(scores) / len(scores)
    else:
        return 0


def get_swn_score_phrase(phrase):
    tokens = word_tokenize(phrase.lower())
    scores = []

    for token in tokens:
        scores.append(get_swn_score(token))

    if scores:
        return sum(scores) / len(scores)
    else:
        return 0
