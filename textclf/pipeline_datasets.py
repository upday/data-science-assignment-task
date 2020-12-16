import lightgbm as lgb
import pandas as pd

import scipy.stats as stats
from sklearn.model_selection import train_test_split

from textclf import PATH_PREPROCESSER, PATH_ML_MODEL
from textclf import DATA_DIR, MODELS_DIR
from textclf import PATH_DATA_INPUT, PATH_DATA_TRAIN_RAW, PATH_DATA_TEST_RAW
from textclf import PATH_DATA_TRAIN_PROCESSED, PATH_DATA_TEST_PROCESSED

from textclf.preprocessing.preprocessing_unit import Preprocesser
from textclf.trainer.trainer import Trainer

from textclf.utils.dataset_split import create_datasets_train_test_raw

import argparse

import logging

logger = logging.getLogger(__name__)


def creating_datasets(path_data, path_train, path_test):

    log_msg = f"loading data from {path_data} ..."
    logger.info(log_msg)
    print(log_msg)
    df = pd.read_csv(path_data, sep="\t")

    log_msg = f"splitting train / test ..."
    logger.info(log_msg)
    print(log_msg)
    df_train_raw, df_test_raw = create_datasets_train_test_raw()

    # ------------------------------------

    log_msg = f"saving dataset train in {path_train} ..."
    logger.info(log_msg)
    print(log_msg)
    df_train_raw.to_csv(path_train, index=False, sep="\t")

    log_msg = f"saving dataset train in {path_test} ..."
    logger.info(log_msg)
    print(log_msg)
    df_test_raw.to_csv(path_test, index=False, sep="\t")

    print("FINISHED")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_data",
        help="path of the file with all data to split",
        action="store",
        type=str,
        default=PATH_DATA_INPUT,
    )
    parser.add_argument(
        "--path_train",
        help="path of the file to use for training",
        action="store",
        type=str,
        default=PATH_DATA_TRAIN_RAW,
    )
    parser.add_argument(
        "--path_test",
        help="path of the file to use for testing",
        action="store",
        type=str,
        default=PATH_DATA_TEST_RAW,
    )

    args = parser.parse_args()

    print(args)

    print(f"args.path_data: {args.path_data}")
    print(f"args.path_train: {args.path_train}")
    print(f"args.path_test: {args.path_test}")

    creating_datasets(args.path_data, args.path_train, args.path_test)
