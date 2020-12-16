import pandas as pd
from sentence_transformers import SentenceTransformer

from textclf.preprocessing.traditional_text_preprocessing import _truncate
from textclf.utils.get_nlp_models import get_bert_model

import logging

logger = logging.getLogger(__name__)


def embedding_text_column_bert(
    text_column: "pd.Series[str]",
    bert_model: "sentence_transformers.SentenceTransformer.SentenceTransformer" = None,
) -> "pd.Series[numpy.ndarray]":

    if bert_model is None:
        bert_model = get_bert_model()

    text_column = _truncate(text_column)

    encodings = bert_model.encode(text_column.tolist())

    return pd.Series(encodings.tolist())


if __name__ == "__main__":

    # distilbert is light and fast
    bert_model = SentenceTransformer("distilbert-base-nli-mean-tokens")

    text_column = pd.Series(['this is an example"'])

    text_column_embedding = embedding_text_column_bert(text_column, bert_model)

    print(text_column_embedding)
    print(text_column_embedding[0].shape)
