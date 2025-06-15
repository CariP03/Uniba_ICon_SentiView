import pandas as pd
from lexicon import create_lexicon

def train_rule_classifier(train_df: pd.DataFrame):
    create_lexicon(train_df)
