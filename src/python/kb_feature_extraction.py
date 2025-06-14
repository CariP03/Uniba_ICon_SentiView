import pandas as pd

from data_load import read_dataset
from tokenizer import clean_tokenize
from prolog_query import compute_metrics_prolog
import config as cfg


df = read_dataset(cfg.DATASET_PATH)

features = []
for text in df["text"]:
    tokens = clean_tokenize(text)
    metrics = compute_metrics_prolog(tokens)
    features.append(metrics)

# Create a dataframe from features
feature_df = pd.DataFrame(
    features,
    columns=["sentiment_sum", "non_zero_count", "num_tokens", "avg_nonzero", "ratio"]
)

df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
df.to_csv(cfg.DATAFRAME_SAVE_PATH, index=False)
