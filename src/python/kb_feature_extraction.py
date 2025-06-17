import os
import pandas as pd

from data_load import read_dataset
from tokenizer import normalized_tokenizer
from prolog_query import compute_metrics_prolog
import config as cfg

def extract_features(df: pd.DataFrame, save=True, save_filename=None) -> pd.DataFrame:

    features = []
    for text in df["text"]:
        tokens = normalized_tokenizer(text)
        metrics = compute_metrics_prolog(tokens)
        features.append(metrics)

    # Create a dataframe from features
    feature_df = pd.DataFrame(
        features,
        columns=["sentiment_sum", "non_zero_count", "num_tokens", "avg_nonzero", "ratio"]
    )

    df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

    if save:
        if save_filename is None:
            save_path = cfg.DATAFRAME_SAVE_PATH / "kb_dataframe.csv"  # default file name
        else:
            save_path = cfg.DATAFRAME_SAVE_PATH / save_filename

        os.makedirs(save_path.parent, exist_ok=True)  # create folder if it does not exist
        df.to_csv(save_path, index=False)

    return df
