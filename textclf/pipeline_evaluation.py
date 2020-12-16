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


def evaluation(path_data_test):

    try:
        log_msg = f"loading test data at {path_data_test} ..."
        print(log_msg)
        logger.info(log_msg)
        df_test_raw = pd.read_csv(path_data_test, sep="\t")
    except Exception as e:
        logger.error(
            f"error when loading df from {path_data_test}, provide correct path for input"
        )
        logger.exception(e)
        raise e

    try:
        logger.info(f"loading previous fitted model {PATH_OVERALL_MODEL} ...")
        with open(PATH_OVERALL_MODEL, "rb") as f:
            m = pickle.load(f)
    except Exception as e:
        logger.error(
            f"error when loading previous fitted model {PATH_OVERALL_MODEL}, launch pipeline_training.py first!"
        )
        logger.exception(e)
        raise e

    logger.info("evaluating df_test_raw ...")
    m.evaluate_model(df_test_raw)

    print("FINISHED")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_data",
        help="path of the file to use for evaluation",
        action="store",
        type=str,
        default=PATH_DATA_TEST_RAW,
    )

    args = parser.parse_args()

    evaluation(args.path_data)
