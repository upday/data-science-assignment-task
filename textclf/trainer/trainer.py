import numpy as np
import pandas as pd

from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.multioutput import MultiOutputClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
import sklearn

from tempfile import mkdtemp
from shutil import rmtree

from textclf.trainer.utils_trainer import *

import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        list_cols_x,
        list_cols_y,
        list_cols_tfidf,
        list_cols_vanilla,
        numeric_features=None,
        categorical_features=None,
        param_dist=None,
    ):
        self.list_cols_x = list_cols_x
        self.list_cols_y = list_cols_y
        self.list_cols_tfidf = list_cols_tfidf
        self.list_cols_vanilla = list_cols_vanilla
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        if numeric_features is None:
            self.numeric_features = ["X_num_ips", "X_num_urls", "X_num_queries"]
        if categorical_features is None:
            self.categorical_features = []

    def get_fitted_pipeline(
        self,
        df,
        n_iter=10,
        param_dist=None,
        tuple_classifier=None,
        scoring="f1_weighted",
        use_memory=False,
        is_easy=True,
    ):

        list_cols_x = self.list_cols_x
        list_cols_tfidf = self.list_cols_tfidf
        list_cols_y = self.list_cols_y
        numeric_features = self.numeric_features
        categorical_features = self.categorical_features

        logger.info(f"I am using {df.shape[0]} records")

        if (param_dist is None) and (tuple_classifier is None):
            tuple_classifier, param_dist = get_default_config(is_easy)

        if len(list_cols_y) > 1:
            raise Exception("I cannot handle len(list_cols_y) > 1 \t for now...")

        df[list_cols_tfidf] = df[list_cols_tfidf].astype(str)

        df = df.sort_index(axis=1)

        df_X = df[list_cols_x]
        df_y = df[list_cols_y].values.ravel()

        df_X = df_X.sort_index(axis=1)

        ###########################################################################

        # We create the preprocessing pipelines for both numeric and categorical data.
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # IMPROVEMENT: handle this case...
        if len(self.list_cols_tfidf) > 1:
            log_msg = (
                "I cannot handle a list_cols_tfidf with more than 1 element."
                "As a workaround concatenate all columns where you want to do the tfidf."
                f"I will take just the first element of the list: {self.list_cols_tfidf[0]}"
            )
            logger.warning(log_msg)

        tfidf_features = self.list_cols_tfidf[0]
        counter_features = self.list_cols_tfidf[0]

        tfidf_transformer = Pipeline(
            steps=[("tfidf_vec", TfidfVectorizer(max_features=100))]
        )

        counter_transformer = Pipeline(steps=[("counter_vec", CountVectorizer())])

        preprocessing = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
                ("tfidf", tfidf_transformer, tfidf_features),
                ("counter_vec", counter_transformer, counter_features),
            ],
            remainder="passthrough",
        )

        # NOTE: I want to cache operations of column transformers
        # "Caching the transformers is advantageous when fitting is time consuming." from documentation
        # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        cachedir = mkdtemp()

        if use_memory:
            logger.info("cache in hypertuning is active")
            memory = cachedir
        else:
            memory = None

        pipe = Pipeline(
            [("preprocessing", preprocessing), tuple_classifier], memory=memory
        )

        # use all cores except one

        import multiprocessing

        n_jobs = multiprocessing.cpu_count()
        if n_jobs > 1:
            n_jobs = n_jobs - 1
        logger.info(f"n_jobs: {n_jobs}")

        # cross validation is done for hyper tuning during randomized search
        cv = StratifiedKFold(n_splits=4)

        # NEVER USE GRID SEARCH
        best = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
            verbose=1,
            refit=True,  # Refit an estimator using the best found parameters on the whole dataset.
            cv=cv,
            return_train_score=True,
        )

        # since refit=True model is trained over all dataset after discover best params
        logger.info("fitting over Train Dataset")

        logger.debug(f"df_X.columns.tolist(): {df_X.columns.tolist()}")

        logger.info(f"df_X.shape: {df_X.shape}")

        try:
            best.fit(df_X, df_y)
        except Exception as e:
            logger.error("error in fit")
            logger.exception(e)

            raise e

        score_cv_train = best.best_score_

        # delete temporary dir I used for cache
        rmtree(cachedir)

        logger.info(f"best.best_params_\n{best.best_params_}")
        logger.info(f"score_cv_train: {score_cv_train}")

        logger.info(f"best.cv_results_\n{best.cv_results_}")

        return best, score_cv_train
