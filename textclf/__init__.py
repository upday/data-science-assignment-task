import logging
import os
from datetime import datetime

PKG_DIR = os.path.abspath("")  # /notebooks
DATA_DIR = os.path.join(PKG_DIR, "data")  # /data
MODELS_DIR = os.path.join(PKG_DIR, "models")  # /models

PATH_DATA_INPUT = os.path.join(DATA_DIR, "data_redacted.tsv")

PATH_DATA_TRAIN_RAW = os.path.join(DATA_DIR, "dataset_train_raw.tsv")
PATH_DATA_TEST_RAW = os.path.join(DATA_DIR, "dataset_test_raw.tsv")

PATH_DATA_TRAIN_PROCESSED = os.path.join(DATA_DIR, "dataset_train_processed.p")
PATH_DATA_TEST_PROCESSED = os.path.join(DATA_DIR, "dataset_train_processed.p")

PATH_DATA_OUTPUT = os.path.join(DATA_DIR, "output.tsv")

PATH_PREPROCESSER = os.path.join(MODELS_DIR, "preprocessor.p")
PATH_ML_MODEL = os.path.join(MODELS_DIR, "ml_model.p")
PATH_OVERALL_MODEL = os.path.join(MODELS_DIR, "m.p")

# stopwords are not avalable after a simple nltk install
import nltk

nltk.download("stopwords")

# https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "false"

##############################################################################
# LOGGER INITIALIZATION AND CONFIGURATION BELOW


def init_logger(config=None, log_level=logging.INFO):
    """
    config can be a filename or a dict and use logging.fileConfig() or logging.dictConfig() accordingly
    """
    if config is not None:
        if type(config) is str:
            logging.config.fileConfig(config)
        elif type(config) is str:
            logging.config.dictConfig(config)
        else:
            __base_config(log_level)

    else:
        __base_config(log_level)


def __base_config(log_level):
    logs_dir = os.path.join("logs", datetime.now().strftime("logs_%Y%m%d"))
    log_name = datetime.now().strftime("log_%Y%m%dT%H%M%S.log")
    os.makedirs(logs_dir, exist_ok=True)
    # create logger with 'spam_application'
    logger = logging.getLogger()
    logger.setLevel(log_level)
    fh = logging.FileHandler(os.path.join(logs_dir, log_name))
    fh.setLevel(log_level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s:%(lineno)s-%(levelname)s-%(message)s threadId(%(thread)d)",
        datefmt="%d/%m/%Y %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    # logging.basicConfig(filename=os.path.join(logs_dir, log_name), level=log_level, format='%(asctime)s-%(name)s:%(lineno)s-%(levelname)s-%(message)s threadId(%(thread)d)', datefmt='%d/%m/%Y %H:%M:%S')
    logging.info("Created logger, level : {:}".format(log_level))


init_logger()
