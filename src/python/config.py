DATASET_PATH = "../../data/aclImdb/train/"
LEXICON_PATH = "../prolog/lexicon.pl"
SENTIMENT_PATH = "../prolog/sentiment_rules.pl"

DATAFRAME_SAVE_PATH = "../../data/kb_train.csv"

# list of intensifiers
INTENSIFIERS = {
    "very": 1.5,
    "extremely": 2.0,
    "really": 1.3,
    "so": 1.2,
    "quite": 1.2,
    "too": 1.2,
    "absolutely": 1.7,
    "highly": 1.5,
    "incredibly": 1.8,
    "super": 1.7,
    "ultra": 1.7
}

# list of negators
NEGATORS = {"not", "never", "no", "hardly", "barely"}

THRESHOLD = 0.1

