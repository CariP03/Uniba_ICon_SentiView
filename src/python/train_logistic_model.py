import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from data_load import read_dataset
from kb_feature_extraction import extract_features
from tokenizer import normalized_tokenizer
import config as cfg


def merge_train_test():
    df_train = read_dataset(cfg.TRAIN_DATASET_PATH)
    df_test = read_dataset(cfg.TEST_DATASET_PATH)

    df = pd.concat([df_train, df_test], ignore_index=True)
    return df


def create_data_splits_with_kb_features(df: pd.DataFrame):
    # split 60% train, 20% val, 20% test
    df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    df_train, df_val = train_test_split(df_train_val, test_size=0.25, random_state=42, stratify=df_train_val["label"])

    df_train = extract_features(df_train, "df_train.csv")
    df_val = extract_features(df_val, "df_val.csv")
    df_test = extract_features(df_test, "df_test.csv")

    return df_train, df_val, df_test


def split_feature_target(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def create_pipeline():
    vectorizer = TfidfVectorizer(
        tokenizer=normalized_tokenizer,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        max_features=10000,
        sublinear_tf=True
    )

    preprocessor = ColumnTransformer(transformers=[
        ("tfidf", vectorizer, "text"),
        ("kbnum", StandardScaler(), ["sentiment_sum", "avg_nonzero", "ratio"])
    ], remainder="drop")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    return pipeline


