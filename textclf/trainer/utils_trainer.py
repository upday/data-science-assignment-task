import numpy as np
import pandas as pd

from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.datasets import load_wine
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
import sklearn

from tempfile import mkdtemp
from shutil import rmtree


def deal_with_stratification(df, list_cols_y, thr, option="duplicate", n=4):
    """[take as input a DataFrame and a target column and decide how to deal with low represented classes:
        depending on the value of 'option' you can either get rid of them or duplicate them.
        Helping with stratify in train_test_split]

    Arguments:
        df {[DataFrame]} -- [input DataFrame]
        list_cols_y {[list]} -- [list with only one element: target column]
        thr {[int]} -- [how few representatives needed to consider a class poorly represented]
        option {[str]} -- [{duplicate|drop}]

    Returns:
        [tDataFrameype] -- [output DataFrame]
    """
    value = list_cols_y[0]
    df_counts = df[value].value_counts().to_frame().reset_index()
    df_counts.columns = ["value", "counts"]
    list_low_counts = df_counts[df_counts.counts <= thr].value.tolist()
    list_high_counts = df_counts[df_counts.counts > thr].value.tolist()
    df_low_counts = df[df[value].isin(list_low_counts)]
    if option == "drop":
        df_output = df[df[value].isin(list_high_counts)]
    elif option == "duplicate":
        df_low_counts = pd.concat([df_low_counts] * (n - 1), ignore_index=True, axis=0)
        df_output = pd.concat([df, df_low_counts], axis=0)

    df_output = df_output.reset_index(drop=True)
    return df_output


# COUNT VECTORIZER
# max_df
# float in range [0.0, 1.0] or int, default=1.0
#     When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
# min_df float in range [0.0, 1.0] or int, default=1
#     When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
# max_features int or None, default=None
#     If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
#     This parameter is ignored if vocabulary is not None.


def _get_default_config_hard():

    param_dist = dict(
        preprocessing__tfidf__tfidf_vec__max_features=stats.randint(1000, 5000),
        preprocessing__tfidf__tfidf_vec__max_df=stats.uniform(0.7, 0.3),
        preprocessing__tfidf__tfidf_vec__min_df=stats.randint(0, 10),
        preprocessing__counter_vec__counter_vec__max_features=stats.randint(100, 200),
        sclf__rf__n_estimators=[100],
        sclf__gbt__n_estimators=[100],
    )

    estimators = [
        ("rf", RandomForestClassifier()),
        ("gbt", GradientBoostingClassifier())
        # ('svc_rbf', SVC(kernel='rbf'))
        # ('nb', MultinomialNB(fit_prior=True, class_prior=None))
        # ('lr', LogisticRegression()),
        # ('knc', KNeighborsClassifier(n_neighbors=3))
    ]

    sclf = StackingClassifier(
        estimators=estimators, final_estimator=SVC(kernel="linear", probability=True)
    )

    tuple_classifier = ("sclf", sclf)

    return tuple_classifier, param_dist


def _get_default_config_soft():

    param_dist = dict(
        preprocessing__tfidf__tfidf_vec__max_features=stats.randint(100, 1000),
        preprocessing__tfidf__tfidf_vec__max_df=stats.uniform(0.7, 0.3),
        preprocessing__tfidf__tfidf_vec__min_df=stats.randint(0, 10),
        preprocessing__counter_vec__counter_vec__max_features=stats.randint(100, 200),
        rf__n_estimators=[500],
    )

    tuple_classifier = ("rf", RandomForestClassifier())

    return tuple_classifier, param_dist


def get_default_config(is_easy=True):
    if is_easy:
        return _get_default_config_soft()
    else:
        return _get_default_config_hard()
