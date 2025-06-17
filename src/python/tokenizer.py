from nltk.corpus import wordnet as wn
import contractions
import spacy

from config import INTENSIFIERS, NEGATORS

nlp = spacy.load("en_core_web_sm")


def tokenize_dataframe(df):
    vocab = set()
    for text in df["text"]:
        tokens = normalized_tokenizer(text)
        for token in tokens:
            vocab.add(token)

    return vocab


# source: https://stackoverflow.com/questions/17245123/getting-adjective-from-an-adverb-in-nltk-or-other-nlp-library
# author: alvas (accessed 13 June 2025)
def convert_adv_to_adj(adverb):
    for ss in wn.synsets(adverb, pos=wn.ADV):
        for lemmas in ss.lemmas():  # all possible lemmas
            pert = lemmas.pertainyms()  # all possible pertainyms
            if pert:
                return pert[0].name()

    return None


def normalized_tokenizer(text: str):
    def get_negated_indices(parsed_text):
        negated = set()
        for tok in parsed_text:
            for child in tok.children:
                if child.dep_ == "neg":
                    negated.add(tok.i)

                    # propagate negation
                    for desc in tok.subtree:
                        if desc.pos_ in {"ADJ", "VERB", "ADV"} and desc.i != tok.i:
                            negated.add(desc.i)
        return negated

    text = contractions.fix(text).lower()
    doc = nlp(text)

    negated_indices = get_negated_indices(doc)

    normalized_tokens = []
    pending_neg = False
    for token in doc:
        if not token.is_alpha:
            continue
        lower = token.text.lower()

        # manage negators that are not identified by spaCy
        if lower in NEGATORS:
            pending_neg = True
            continue

        # intensifiers
        if lower in INTENSIFIERS:
            normalized_tokens.append(lower)
            continue

        # lemmatize
        lemma = token.lemma_.lower()
        # convert from adverb to adjective
        if token.pos_ == "ADV":
            adj = convert_adv_to_adj(lower)
            if adj:
                lemma = adj

        # apply negation
        if token.i in negated_indices or pending_neg:
            lemma = f"not_{lemma}"
            pending_neg = False

        normalized_tokens.append(lemma)

    return normalized_tokens