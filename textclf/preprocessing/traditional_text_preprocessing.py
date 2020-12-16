import string
import spacy
import pandas as pd
import numpy as np
from functools import partial

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

from textclf.utils.get_nlp_models import get_spacy_model

from textclf.utils.parallel_functions import parallelize_on_rows

import logging

logger = logging.getLogger(__name__)


def _truncate(column, max_num_tokens=512):

    column = column.astype(str)

    # for each element
    def _truncate_text(text, max_num_tokens):
        return " ".join(text.split()[:max_num_tokens])

    _truncate_text_partial = partial(_truncate_text, max_num_tokens=max_num_tokens)
    return column.apply(_truncate_text_partial)


def _to_lower(column):
    return column.str.lower()


def _remove_punct(column):
    punct = string.punctuation
    punct += "’“”—‘"
    transtab = str.maketrans(dict.fromkeys(punct, ""))
    return column.str.translate(transtab)


def _remove_numbers(column):
    return column.replace("\d+", "", regex=True)


def _remove_stopwords(column):
    # https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python

    # IMPROVEMENT: move this load externally to speed up (_remove_stopwords is called multiple times)
    stop_words = stopwords.words("english")
    stopwords_dict = Counter(stop_words)

    def _filter_tokens(text, min_num_word=4):
        return " ".join(
            [
                word
                for word in text.split()
                if (word not in stopwords_dict) and (len(word) >= min_num_word)
            ]
        )

    return column.apply(_filter_tokens)


def _apply_stemming(column):
    stemmer = PorterStemmer()

    list_text = column.tolist()
    list_text_stemmed = [
        " ".join([stemmer.stem(x) for x in text.split()]) for text in list_text
    ]

    column = pd.Series(list_text_stemmed)
    return column


def _apply_lemmatization(column, spacy_model=None):

    if spacy_model is None:
        spacy_model = get_spacy_model()

    docs = list(spacy_model.pipe(column.tolist(), disable=["tagger", "parser", "new"]))

    list_text_lemm = [" ".join([x.lemma_ for x in doc]) for doc in docs]

    column = pd.Series(list_text_lemm)

    return column


def traditional_text_preprocessing(
    text_column: "pd.Series[str]",
    spacy_model=None,
    lemmatization: bool = False,
    stemming: bool = False,
) -> pd.Series:
    """
    Apply this function if you want to preprocess a text column before using
    traditional ML algorithm like Tf-IDF. Not before Transformer!

    operations performed
        - truncate the number of words
        - to lower case
        - remove punctuation
        - remove stopwords
        - apply stemming / apply lemmatization

    Args:
        df (pd.Series[str]): raw text column of your dataframe (example 'title')

    Returns:
        pd.Series[str]: cleaned text column
    """
    # TODO: adjust docstring

    logger.info(f"lemmatization: {lemmatization}")
    logger.info(f"stemming: {stemming}")

    if lemmatization and stemming:
        logger.warning(
            "lemmatization and stemming are both True! I'll use only lemmatization"
        )

    logger.info(f"_truncate ...")
    text_column = _truncate(text_column)

    logger.info(f"_to_lower ...")
    text_column = _to_lower(text_column)

    logger.info(f"_remove_punct ...")
    text_column = _remove_punct(text_column)

    logger.info(f"_remove_numbers ...")
    text_column = _remove_numbers(text_column)

    logger.info(f"_remove_stopwords ...")
    text_column = _remove_stopwords(text_column)

    if stemming:
        logger.info(f"_apply_stemming ...")
        text_column = _apply_stemming(text_column)

    if lemmatization:
        logger.info(f"_apply_lemmatization ...")
        text_column = _apply_lemmatization(text_column, spacy_model)

    return text_column


if __name__ == "__main__":

    text_column = pd.Series(
        ["this is an example of a new sentence", "adass dasdsasdasda 42 meme triggered"]
    )
    text_column = traditional_text_preprocessing(
        text_column, lemmatization=True, stemming=False
    )

    print(text_column)
