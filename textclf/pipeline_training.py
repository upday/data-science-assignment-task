import lightgbm as lgb
import pandas as pd
import pickle

import scipy.stats as stats
from sklearn.model_selection import train_test_split

from textclf import PATH_PREPROCESSER, PATH_ML_MODEL, PATH_OVERALL_MODEL
from textclf import DATA_DIR, MODELS_DIR
from textclf import PATH_DATA_INPUT, PATH_DATA_TRAIN_RAW, PATH_DATA_TEST_RAW
from textclf import PATH_DATA_TRAIN_PROCESSED, PATH_DATA_TEST_PROCESSED

from textclf.preprocessing.preprocessing_unit import Preprocesser
from textclf.trainer.trainer import Trainer
from textclf.overall_model import ComplexModel

import argparse


import logging

logger = logging.getLogger(__name__)


def training(path_train, use_spacy: bool, use_bert: bool):

    m = ComplexModel(use_spacy=use_spacy, use_bert=use_bert)

    try:
        log_msg = f"loading training data at {path_train} ..."
        print(log_msg)
        logger.info(log_msg)
        df_train_raw = pd.read_csv(path_train, sep="\t")
    except Exception as e:
        logger.error(
            f"error when loading df from {path_train}, provide correct path for input"
        )
        logger.exception(e)
        raise e

    log_msg = f"processing training data ..."
    print(log_msg)
    logger.info(log_msg)
    df_train = m.tranform_df_train(df_train_raw)

    log_msg = f"fitting ml model ..."
    print(log_msg)
    logger.info(log_msg)
    m.get_fitted_model(df_train)

    log_msg = f"saving overall fitted model in {PATH_OVERALL_MODEL} ..."
    print(log_msg)
    logger.info(log_msg)
    with open(PATH_OVERALL_MODEL, "wb") as f:
        pickle.dump(m, f)

    print("FINISHED")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_train",
        help="path of the file to use for training",
        action="store",
        type=str,
        default=PATH_DATA_TRAIN_RAW,
    )
    parser.add_argument(
        "--use_spacy",
        help="enable spacy embeddings",
        dest="use_spacy",
        action="store_true",
    )
    parser.add_argument(
        "--use_bert",
        help="enable bert embeddings",
        dest="use_bert",
        action="store_true",
    )

    args = parser.parse_args()

    training(args.path_train, args.use_spacy, args.use_bert)
