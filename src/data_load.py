import os
import pandas as pd


def read_folder(path: str, positive: bool) -> pd.DataFrame:
    data = []

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            data.append({
                'text': text,
                'label': 1 if positive else 0
            })

    return pd.DataFrame(data)

def read_dataset(path:str) -> pd.DataFrame:
    pos_path = os.path.join(path, 'pos')
    neg_path = os.path.join(path, 'neg')

    df_pos = read_folder(pos_path, True)
    df_neg = read_folder(neg_path, False)

    return pd.concat([df_pos, df_neg])