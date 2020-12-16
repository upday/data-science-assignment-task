import lightgbm as lgb
import pandas as pd

import scipy.stats as stats

from textclf.preprocessing.preprocessing_unit import Preprocesser
from textclf.trainer.trainer import Trainer

from textclf.utils.report_performance import report_performance

import logging

logger = logging.getLogger(__name__)


class ComplexModel:
    def __init__(self, use_spacy: bool = True, use_bert: bool = False):
        """
        This obj can train, predict and evaluate, see other methods for more details.

        Args:
            use_spacy (bool, optional): if True extract spacy embeddings. Defaults to False.
            use_bert (bool, optional): if True extract bert embeddings. Defaults to False.
        """

        # this object briefly performs
        #   - feature extraction
        #   - cleaning
        #   - embedding
        # see comments in methods and attributes of Preprocesser for more info
        self.use_spacy = use_spacy
        self.use_bert = use_bert

        logger.info(f"use_spacy: {use_spacy}")
        logger.info(f"use_bert: {use_bert}")

        self.p = Preprocesser(use_spacy=use_spacy, use_bert=use_bert)

        self.ml_model = None

        self.list_cols_x = []
        self.list_cols_y = []
        self.scoring = None

    def tranform_df_train(self, df_train_raw: pd.DataFrame) -> pd.DataFrame:
        """[summary]

        Args:
            df_train_raw (pd.DataFrame): raw dataset to use as starting point for processing

        Returns:
            pd.DataFrame: processed dataframe
        """

        logger.info("creating dataset ...")
        df_train = self.p.fit_transform(df_train_raw)

        return df_train

    def get_fitted_model(self, df: pd.DataFrame) -> pd.DataFrame:

        list_cols_y = ["category"]  # col to predict
        list_cols_tfidf = ["all_text_clean_concat"]  # where applaying tfidf and/or tf
        list_cols_vanilla = [
            x for x in df.columns if "emb_" in x
        ]  # cols to use without changing anything
        numeric_features = [
            "text_len",
            "title_len",
        ]  # numeric cols --> i rescale them...
        categorical_features = ["netloc"]  # conversion to one-hot
        scoring = (
            "f1_weighted"  # scoring to compare different models in randomized search
        )

        list_cols_x = (
            list_cols_tfidf
            + list_cols_vanilla
            + numeric_features
            + categorical_features
        )
        list_all_cols = list_cols_x + list_cols_y

        # i save the list of columns used for the prediction part
        self.list_cols_x = list_cols_x
        self.list_cols_y = list_cols_y

        # as well as the score system
        self.scoring = scoring

        # obj to coordinates hypertuning and preprocessing with tf-idf, one-hot, ...
        # NOTE: Trainer obj is a sort of wrapper of (randomized search) + (pipeline obj by sklearn)
        t = Trainer(
            list_cols_x=list_cols_x,
            list_cols_y=list_cols_y,
            list_cols_tfidf=list_cols_tfidf,
            list_cols_vanilla=list_cols_vanilla,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        # i sample parameters to test from these distributions

        # FIXME: distributions chosen in an intitive way for now after a few experiments!!!
        param_dist_lgb = dict(
            preprocessing__tfidf__tfidf_vec__max_features=stats.randint(4000, 5000),
            preprocessing__tfidf__tfidf_vec__max_df=stats.uniform(0.9, 0.1),
            preprocessing__tfidf__tfidf_vec__min_df=stats.randint(5, 15),
            preprocessing__counter_vec__counter_vec__max_features=stats.randint(
                1000, 1500
            ),
            preprocessing__counter_vec__counter_vec__max_df=stats.uniform(0.7, 0.3),
            preprocessing__counter_vec__counter_vec__min_df=stats.randint(5, 20),
            lgb__n_estimators=[200],
            lgb__reg_lambda=[0, 1e-1],  # , 1, , 5, 10, 20],
            lgb__reg_alpha=[0, 1e-1],  # , 1, 2, 5, 7, 10],
        )

        # estimator i want to use on top of the preprocessing
        tuple_classifier_lgb = ("lgb", lgb.LGBMClassifier(class_weight="balanced"))

        # take just the cols i need + sort cols by name
        # NOTE: the order of columns must NOT change
        df = df[list_all_cols].sort_index(axis=1)

        # launch the fitting part
        best_pipe_lgb, score_cv_train_lgb = t.get_fitted_pipeline(
            df,
            n_iter=1,
            param_dist=param_dist_lgb,
            tuple_classifier=tuple_classifier_lgb,
            scoring=scoring,
        )

        log_msg = f"{scoring} train: {score_cv_train_lgb}"
        print(log_msg)
        logger.info(log_msg)

        # save ml_model
        self.ml_model = best_pipe_lgb

    def predict(self, df_raw: pd.DataFrame) -> "np.ndarray":
        """
        given a dataframe as input predict the categories

        Args:
            df_raw (pd.DataFrame): dataframe raw

        Returns:
            np.ndarray: "category" predictions
        """

        logger.info(f"transforming df_test_raw ...")
        df = self.p.transform(df_raw)

        # select just the cols I need...
        X = df[self.list_cols_x]

        # sort columns alphabetically
        X = X.sort_index(axis=1)

        logger.info(f"predicting ...")
        y_pred = self.ml_model.predict(X)

        return y_pred

    def evaluate_model(self, df_test_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate model giving a dataset with bot X and y cols.
        A performance report is plotted

        Args:
            df_test_raw (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """

        logger.info(f"predicting ...")
        y_pred = self.predict(df_test_raw)
        y_test = df_test_raw[self.list_cols_y]

        logger.info(f"computing performances ...")
        report_performance(y_test, y_pred)
