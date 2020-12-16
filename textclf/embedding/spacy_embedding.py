import pandas as pd
import spacy
from functools import partial


from textclf.preprocessing.traditional_text_preprocessing import _truncate
from textclf.utils.parallel_functions import parallelize_on_rows
from textclf.utils.get_nlp_models import get_spacy_model

import logging

logger = logging.getLogger(__name__)


def embedding_text_column_spacy(
    text_column: "pd.Series[str]", nlp_model: "spacy.lang.en.English" = None
) -> "pd.Series[numpy.ndarray]":

    """
    Use a standard LM like spacy to extarct the sentence embedding.
    In this case is just the mean of the embeddings of the words.
    Spacy apply a sort of w2v to each word

    Args:
        text_column (pd.Series[str]): text column, or in other words, a series containing strings
        nlp_model (spacy.lang.en.English): spacy model
    Returns:
        pd.Series[numpy.ndarray]: column where each element is a np array of the embedding
    """

    if nlp_model is None:
        nlp_model = get_spacy_model()

    text_column = _truncate(text_column).astype(str)

    # https://spacy.io/usage/processing-pipelines
    docs = list(
        nlp_model.pipe(text_column.tolist(), disable=["tagger", "parser", "new"])
    )

    # for each doc in docs i take .vector
    # that is the mean vector for the entire sentence

    return pd.Series([x.vector for x in docs])


if __name__ == "__main__":

    nlp_model = spacy.load("en_core_web_lg")

    text_column = pd.Series(["this is an example"])
    text_column_embedding = embedding_text_column_spacy(text_column, nlp_model)

    another_text_column = pd.Series(["another sentence just to consider"])
    another_text_column_embedding = embedding_text_column_spacy(
        another_text_column, nlp_model
    )

    print(text_column_embedding)
    print(text_column_embedding[0].shape)
