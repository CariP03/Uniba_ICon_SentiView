from sentiwordnet_score import get_swn_score
from tokenizer import tokenize_dataframe
from data_load import read_dataset


DATASET_PATH = "..\\..\\data\\aclImdb\\train\\"
LEXICON_PATH = "..\\prolog\\lexicon.pl"

THRESHOLD = 0.1
INTENSIFIERS = [["extremely", 2.0], ["very", 1.5], ["really", 1.3], ["so", 1.2], ["quite", 1.2]]
NEGATORS = ["not", "no", "hardly", "barely", "never"]

with open(LEXICON_PATH, "w", encoding="utf-8") as lexicon:
    vocab = tokenize_dataframe(read_dataset(DATASET_PATH))

    # calculate a score for each token
    for token in vocab:
        score = get_swn_score(token)

        if abs(score) >= THRESHOLD:
            safe_token = token.replace('"', '')
            lexicon.write(f'word_score("{safe_token}", {score:.4f}).\n')

    # write intensifiers
    lexicon.write('\n% Intensifiers\n')
    for intensifier in INTENSIFIERS:
        lexicon.write(f'intensifier("{intensifier[0]}", {intensifier[1]}).\n')

    # write negators
    lexicon.write('\n% Negators\n')
    for negator in NEGATORS:
        lexicon.write(f'negator("{negator}").\n')