import os
from pyswip import Prolog

import config as cfg
from tokenizer import normalized_tokenizer


def load_kb():
    # load knowledge base
    prolog = Prolog()

    lex_path = os.path.abspath(cfg.LEXICON_PATH).replace('\\', '/')
    list(prolog.query(f"consult('{lex_path}')"))

    rule_path = os.path.abspath(cfg.SENTIMENT_PATH).replace('\\', '/')
    list(prolog.query(f"consult('{rule_path}')"))

    # reset token_at
    list(prolog.query("retractall(token_at(_, _))"))

    return prolog

def compute_metrics_prolog(tokens):
    prolog = load_kb()

    # assert all tokens
    for index, token in enumerate(tokens, start=1):
        prolog.assertz(f'token_at({index}, "{token}")')

    # query the knowledge base
    sum_query = list(prolog.query("sentiment_sum(S)"))
    cnt_query = list(prolog.query("sentiment_count_nonzero(C)"))
    tot_query = list(prolog.query("sentiment_total_tokens(T)"))

    # retrieve results
    sentiment_sum = sum_query[0]['S'] if sum_query else 0
    non_zero_count = cnt_query[0]['C'] if cnt_query else 0
    num_tokens = tot_query[0]['T'] if tot_query else 0

    # calculate useful metrics
    avg_nonzero = sentiment_sum / non_zero_count if non_zero_count > 0 else 0
    ratio = non_zero_count / num_tokens if num_tokens > 0 else 0

    return sentiment_sum, non_zero_count, num_tokens, avg_nonzero, ratio

def classify_review(review: str) -> bool:
    prolog = load_kb()

    # assert all tokens
    tokens = normalized_tokenizer(review)
    for index, token in enumerate(tokens, start=1):
        prolog.assertz(f'token_at({index}, "{token}")')

    # query the knowledge base
    label = list(prolog.query("is_positive"))
    return bool(label)
