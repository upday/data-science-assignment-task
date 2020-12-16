import lightgbm as lgb
import pandas as pd
import pickle

import scipy.stats as stats

from textclf.preprocessing.preprocessing_unit import Preprocesser
from textclf.utils.dataset_split import create_datasets_train_test_raw
from textclf.trainer.trainer import Trainer

from textclf.overall_model import ComplexModel

from textclf.utils.report_performance import report_performance

import logging

logger = logging.getLogger(__name__)


def main():

    use_spacy = False
    use_bert = False
    m = ComplexModel(use_spacy=use_spacy, use_bert=use_bert)

    df_train_raw, df_test_raw = create_datasets_train_test_raw()

    df_train = m.tranform_df_train(df_train_raw)

    m.get_fitted_model(df_train)

    m.evaluate_model(df_test_raw)

    with open("m.p", "wb") as f:
        pickle.dump(m, f)

    return


if __name__ == "__main__":

    main()
