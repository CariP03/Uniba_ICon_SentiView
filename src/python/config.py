from pathlib import Path

# find project root
CONFIG_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_PATH.parent.parent.parent

TRAIN_DATASET_PATH = PROJECT_ROOT / "data" / "aclImdb" / "train"
TEST_DATASET_PATH = PROJECT_ROOT / "data" / "aclImdb" / "test"
DATAFRAME_SAVE_PATH = PROJECT_ROOT / "data"
MODEL_SAVE_PATH = PROJECT_ROOT / "models"

LEXICON_PATH = PROJECT_ROOT / "src" / "prolog" / "lexicon.pl"
SENTIMENT_PATH = PROJECT_ROOT / "src" / "prolog" / "sentiment_rules.pl"

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
NEGATORS = {"not", "never", "no", "hardly", "barely", "little", "scarcely"}

THRESHOLD = 0.1

