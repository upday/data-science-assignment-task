import spacy
from sentence_transformers import SentenceTransformer

import logging

logger = logging.getLogger(__name__)


def get_bert_model():

    # list pretrained models
    # https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0

    # distilbert is light and fast
    # default_model = 'distilbert-base-nli-mean-tokens'
    default_model = "distilbert-base-nli-stsb-wkpooling"
    logger.warning(f"loading defualt {default_model} transformer model")

    bert_model = SentenceTransformer(default_model)

    return bert_model


def get_spacy_model():

    # link with possible spacy models i can use
    # https://spacy.io/models/en/
    defualt_model = "en_core_web_lg"  # large
    defualt_model = "en_core_web_md"  # medium

    # NOTE: i cant use small model...
    # from https://spacy.io/usage/vectors-similarity:
    # "[...] To make them compact and fast, spaCy’s small models (all packages that end in sm) don’t ship with word vectors,
    # and only include context-sensitive tensors. This means you can still use the similarity()
    # methods to compare documents, spans and tokens – but the result won’t be as good, and individual tokens won’t have any vectors assigned.
    # So in order to use real word vectors, you need to download a larger model [...]"

    logger.warning(f"loading defualt {defualt_model} spacy model")
    nlp_model = spacy.load(defualt_model)

    return nlp_model
