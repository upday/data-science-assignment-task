import pandas as pd

from sklearn.model_selection import train_test_split

from textclf import PATH_DATA_INPUT

import logging

logger = logging.getLogger(__name__)


def create_datasets_train_test_raw() -> None:
    """
    Split the dataset in train and test paying attention to "category" distribution.
    Datasets are then saved in PATH_DATA_TRAIN_RAW and PATH_DATA_TEST_RAW

    Returns:
        None
    """

    df = pd.read_csv(PATH_DATA_INPUT, "\t")

    # split train test with stratify to better balances class distributions
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["category"]
    )

    logger.info(f"I am using {df_train.shape[0]} \t TRAIN records")
    logger.info(f"I am using {df_test.shape[0]} \t TEST records")

    # df_train.to_csv(PATH_DATA_TRAIN_RAW, sep='\t', index=False)
    # df_test.to_csv(PATH_DATA_TEST_RAW, sep='\t', index=False)

    return df_train, df_test


if __name__ == "__main__":

    create_datasets_train_test_raw()
