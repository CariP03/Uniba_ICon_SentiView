from sentiwordnet_score import get_swn_score
from tokenizer import tokenize_dataframe

from data_load import read_dataset
import config as cfg

with open(cfg.LEXICON_PATH, "w", encoding="utf-8") as lexicon:
    vocab = tokenize_dataframe(read_dataset(cfg.DATASET_PATH))

    # calculate a score for each token
    for token in vocab:
        score = get_swn_score(token)

        if abs(score) >= cfg.THRESHOLD:
            lexicon.write(f'word_score("{token}", {score:.4f}).\n')

    # write intensifiers
    lexicon.write('\n% Intensifiers\n')
    for intensifier in cfg.INTENSIFIERS:
        lexicon.write(f'intensifier("{intensifier}", {cfg.INTENSIFIERS[intensifier]}).\n')

    # write negators
    lexicon.write('\n% Negators\n')
    for negator in cfg.NEGATORS:
        lexicon.write(f'negator("{negator}").\n')