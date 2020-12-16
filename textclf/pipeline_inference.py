import pandas as pd
import pickle


from textclf import PATH_OVERALL_MODEL
from textclf import PATH_DATA_TEST_RAW
from textclf import PATH_DATA_OUTPUT


import argparse


import logging

logger = logging.getLogger(__name__)


def inference(path_data, path_result):

    try:
        log_msg = f"loading test data at {path_data} ..."
        print(log_msg)
        logger.info(log_msg)
        df_raw = pd.read_csv(path_data, sep="\t")
    except Exception as e:
        logger.error(
            f"error when loading df from {path_train}, provide correct path for input"
        )
        logger.exception(e)
        raise e

    print(f"predicting {df_raw.shape[0]} categories")
    logger.info(f"predicting {df_raw.shape[0]} categories")

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
    y_pred = m.predict(df_raw.copy())

    df_raw_pred = df_raw.copy()
    df_raw_pred["category_prediction"] = y_pred

    df_raw_pred.to_csv(path_result, index=False, sep="\t")

    print("FINISHED")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_data",
        help="path of the file with data to predict",
        action="store",
        type=str,
        default=PATH_DATA_TEST_RAW,
    )
    parser.add_argument(
        "--path_result",
        help="path of the file with prediction",
        action="store",
        type=str,
        default=PATH_DATA_OUTPUT,
    )

    args = parser.parse_args()

    inference(args.path_data, args.path_result)
