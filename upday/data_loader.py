"""
This module contains two classes that encapsulate data loading an model saving.
For this assignment, these class use the file-system and their behaviour is
hard-coded. In a production setup, we would use implementations of these
interfaces that would likely setup a database connection.

One might wonder if the DataLoader.load method should take a query argument.
However, that would make assumptions about the specific type of data base
backend (e.g. accepting SQL queries). For our usecase, there will only ever be
a single dataset for training. Thus, it is sufficient to only allow one dataset
to be loaded.
"""
import pickle
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, seed=4381, test_fraction=0.3):
        self.seed = seed
        self.test_fraction = test_fraction
        self.__indices = None
        self.__data = None

    def load(self, subset='train'):
        if self.__data is None:
            filename = './data/data_redacted.tsv'
            self.__data = pd.read_csv(filename, sep='\t')

        test_size = int(self.test_fraction * len(self.__data))
        self.__indices = np.arange(len(self.__data))
        np.random.seed(self.seed)
        np.random.shuffle(self.__indices)

        if subset.lower() == 'train':
            return self.__data.iloc[self.__indices[test_size:]]
        elif subset.lower() == 'test':
            return self.__data.iloc[self.__indices[:test_size]]
        else:
            raise ValueError(
                f'Invalid subset "{subset}".'
                ' Should be either "train" or "test"'
            )


class ModelSaver:
    def store(self, model):
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

    def load(self):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)