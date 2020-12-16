import pandas as pd
import numpy as np
import re
import os
import pickle
from functools import partial
from urllib.parse import urlparse


from textclf.preprocessing.traditional_text_preprocessing import (
    traditional_text_preprocessing,
)
from textclf.embedding.spacy_embedding import embedding_text_column_spacy
from textclf.embedding.bert_embedding import embedding_text_column_bert
from textclf.utils.get_nlp_models import get_spacy_model, get_bert_model

import logging

logger = logging.getLogger(__name__)


class Preprocesser:
    def __init__(self, use_spacy: bool = False, use_bert: bool = False):
        """
        this object will take care about preprocessing before the randomized search of the ml_model.


        Args:
            use_spacy (bool, optional): if True extract spacy embeddings. Defaults to False.
            use_bert (bool, optional): if True extract bert embeddings. Defaults to False.
        """

        self.fictious_label = "Other"
        self.use_spacy = use_spacy
        self.use_bert = use_bert

        self.unique_netlocs = set()
        self.fit = False

        self.spacy_model = get_spacy_model()
        self.bert_model = get_bert_model()

    def _base_preprocessing(
        self, df: pd.DataFrame, predict: bool = False
    ) -> pd.DataFrame:
        """
        Apply a basic processing. in particular
            - extraction of the len of words for "title" and "text" columns
            - extraction of "netloc" and "path" from urls
        Those transformations still hold for both transformer and traditional ML models

        Args:
            df (pd.DataFrame): raw dataframe input (it should have the structure of "data_redacted.tsv")
            predict (bool): Am I predicting? Deafult to False

        Returns:
            pd.DataFrame: preprocessed dataframe
        """

        # add len tokens
        df["title_len"] = df["title"].str.len()
        df["text_len"] = df["text"].str.len()

        # parse all urls
        df["urlparse"] = df["url"].apply(urlparse)

        # extract domain name
        df["netloc"] = df["urlparse"].apply(lambda x: x.netloc)

        # extract path
        df["path"] = df["urlparse"].apply(lambda x: x.path)

        if not predict:
            # I am in this case when I am creating the dataset for the first time
            # i need to find the list of netloc to keep ("self.unique_netlocs")

            # i keep all netlocs that appear with a minimum %
            minimum_frequency_perc = 0.01
            count_enough = (
                df["netloc"].value_counts(normalize=True) > minimum_frequency_perc
            )

            netloc_col_keep = count_enough[count_enough].index
            unique_netlocs = set(netloc_col_keep.unique())
            self.unique_netlocs = unique_netlocs

        # mask that tells where I have the netlocs to keep
        mask = df["netloc"].isin(self.unique_netlocs)

        # while ~mask if for all the netlocs to replace...
        df["netloc"] = np.where(mask, df["netloc"], self.fictious_label)

        # transform "path" into a sort of normal "text"
        #   path is something like 'beauty/How-Use-Brow-Gel-38516588'
        #   after this function will become 'beauty How Use Brow Gel 38516588'
        compiled_regex = re.compile("[-/_.]")

        def _path_to_text(path, compiled_regex):
            return " ".join(compiled_regex.split(path))

        path_to_text_partial = partial(_path_to_text, compiled_regex=compiled_regex)
        df["path"] = df["path"].apply(path_to_text_partial)
        # then i will pass this "path" to the standard cleaning pipeline (like for "text" or "title")

        # i dont need "urlparse" and "url" once i have extracted netloc and path
        df = df.drop(columns=["urlparse", "url"])

        # clean for tf-idf
        df["text_clean"] = traditional_text_preprocessing(
            df["text"], self.spacy_model, lemmatization=False
        )
        df["title_clean"] = traditional_text_preprocessing(
            df["title"], self.spacy_model, lemmatization=False
        )
        df["path_clean"] = traditional_text_preprocessing(
            df["path"], self.spacy_model, lemmatization=False
        )

        # NOTE: it would be interesting to keep those sources sperated
        # anyway, i concatenate the columns to speed up
        df["all_text_clean_concat"] = (
            df["title_clean"] + df["text_clean"] + df["path_clean"]
        )

        return df

    def _add_spacy_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        concatenate a column called "embeddings_spacy" containing a np.array with
        the embedding of each sentence computed by spacy model.
        Embeddings are vectors of 300 elements and is the average of the embedding of each word in a sentence
        (see "embedding_text_column_spacy" for more info)

        Args:
            df (pd.DataFrame): output dataframe of method "_base_preprocessing" containing columns "text_clean" and "title_clean"

        Returns:
            pd.DataFrame: same input dataset with a new column named "embeddings_spacy"
        """

        # compute spacy embeddings for these 2 columns and save results in aux cols
        df["text_clean_emb"] = embedding_text_column_spacy(
            df["text_clean"], self.spacy_model
        )
        df["title_clean_emb"] = embedding_text_column_spacy(
            df["title_clean"], self.spacy_model
        )

        # sum the 2 embeddings
        df["embeddings_spacy"] = df["text_clean_emb"] + df["title_clean_emb"]

        # drop aux embedding cols
        df = df.drop(columns=["text_clean_emb", "title_clean_emb"])

        return df

    def _transformer_text_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        concatenate a column called "embeddings_bert" containing a np.array with
        the embedding of each sentence computed by a transformer model.
        Embeddings are vectors of 768 elements and is the average of the embedding of each word in a sentence
        (see "embedding_text_column_bert" for more info)

        Args:
            df (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """

        logger.info('embedding_text_column_bert "text" ...')
        df["text_embeddings"] = embedding_text_column_bert(df["text"], self.bert_model)

        logger.info('embedding_text_column_bert "title" ...')
        df["title_embeddings"] = embedding_text_column_bert(
            df["title"], self.bert_model
        )

        # sum the 2 embeddings
        df["embeddings_bert"] = df["text_embeddings"] + df["title_embeddings"]

        # drop aux embedding cols
        df = df.drop(columns=["text_embeddings", "title_embeddings"])

        return df

    def _concatenate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        After all transformations and embeddings, i want to handle the columns with embeddings
        in the case they were computed.
        every embedding column contains a ndarray that we cannot give in input to the ml_model.fit method.
        In fact we need only columns with single values.
        So each column is tranformed in a list of columns whose number is equal to the len of the embedding.
        This "spread" operation is done by _spread_column function below

        Example: from a column coming from spacy embedding i extract 300 columns

        Args:
            df (pd.DataFrame): dataframe after all transformations and preprocessings

        Returns:
            pd.DataFrame: [description]
        """

        def _spread_column(df, name_column, prefix):
            s = df[name_column]
            df_tmp = pd.DataFrame.from_dict(dict(zip(s.index, s.values))).T
            df_tmp.columns = [f"{prefix}_{x}" for x in df_tmp.columns]

            df = pd.concat([df, df_tmp], axis=1)
            df = df.drop(columns=[name_column])
            return df

        if "embeddings_spacy" in df.columns:
            df = _spread_column(df, "embeddings_spacy", "emb_spacy_")

        if "embeddings_bert" in df.columns:
            df = _spread_column(df, "embeddings_bert", "emb_bert_")

        return df

    def _complete_preprocessing(
        self, df: pd.DataFrame, predict: bool = False
    ) -> pd.DataFrame:

        """
        method that orchestrates all preprocessing operations.
        Just a note about predict input...

        there are some transformations that are based on the distribution of a particular label
        once that i know the distribution when i am creating the dataset
        i keep and use that info when predicting
        Example: this is the case for netloc --> see _base_preprocessing for more details

        _add_spacy_embeddings and _transformer_text_preprocessing
        are based only on pre-trained models, so they are "independent" from my dataset
        therefore i dont need to specify the predict flag as before

        Args:
            df (pd.DataFrame): input raw dataframe (just after loading the original data in tsv format)
            predict (bool): flag that tells if you are predicting or if you are training otherwise. Deafult False
        Returns:
            df (pd.DataFrame): dataframe processed ready for ml pipeline (aka randomized search in this case)
        """

        df = self._base_preprocessing(df, predict=predict)

        if self.use_spacy:
            df = self._add_spacy_embeddings(df)

        if self.use_bert:
            df = self._transformer_text_preprocessing(df)

        # order columns alphabetically
        df = df.sort_index(axis=1)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call this method when you are training.
        this name is chosen in order to make you recall the standard method of objects in sklearn (like pca).
        So in fit_transform you fit your objects and transform the input

        Args:
            df (pd.DataFrame): raw dataframe loaded from original data (structure of .tsv)

        Returns:
            pd.DataFrame: processed dataframe
        """

        if self.fit:
            log_msg = "Preprocessor was already fit"
            logger.warning(log_msg)

        df = self._complete_preprocessing(df, predict=False)
        self.fit = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call this method when you are only predicting.
        this name is chosen in order to make you recall the standard method of objects in sklearn (like pca).
        So in transform you DON'T fit your objects and only transform the input

        Args:
            df (pd.DataFrame): raw dataframe loaded from original data (structure of .tsv)

        Raises:
            ValueError: if you have never called fit_transform before

        Returns:
            pd.DataFrame: processed dataframe
        """

        if not self.fit:
            log_msg = "Preprocesser tries to transform data even if it has never seen any data before"
            logger.error(log_msg)
            raise ValueError(log_msg)

        df = self._complete_preprocessing(df, predict=True)
        return df


if __name__ == "__main__":

    from textclf import PATH_DATA_INPUT, DATA_DIR

    df = pd.read_csv(PATH_DATA_INPUT, "\t")

    p = Preprocesser()

    df_train = df.head(1000).copy()
    df_train = p.fit_transform(df_train)
    destination_path = os.path.join(DATA_DIR, "dataset_train.p")
    df_train.to_pickle(destination_path)

    with open("preprocessor.p", "wb") as f:
        pickle.dump(p, f)

    with open("preprocessor.p", "rb") as f:
        p = pickle.load(f)

    df_test = df.tail(100).copy()
    df_test = p.transform(df_test, predict=True)
    destination_path = os.path.join(DATA_DIR, "dataset_test.p")
    df_test.to_pickle(destination_path)
