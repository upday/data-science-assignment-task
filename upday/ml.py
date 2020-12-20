from collections import namedtuple
import numpy as np
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class TextEmbedding:
    def __init__(self, url="https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"):
        self.url = url
        # This is in SavedModel 2.0 format trained on google news and maps
        # from text to 50-dimensional embedding vectors.
        # Browse the TF Hub repository (https://tfhub.dev) for more options.
        self.embed = hub.load(url)  # Maybe cache locally?

    def process(self, texts):
        embedded = self.embed(texts)
        return embedded.numpy()

    def __setstate__(self, url):
        self.url = url
        self.embed = hub.load(self.url)

    def __getstate__(self):
        return self.url


CategoryModel = namedtuple('CategoryModel', ['embeddings', 'classifier'])

# Ignore ConvergenceWarning: We don't need the best model as long as it is
# good enough
@ignore_warnings(category=ConvergenceWarning)
def train(data):
    embeddings = TextEmbedding()
    classifier = LogisticRegressionCV()

    X = np.c_[
        embeddings.process(data['title'].values.tolist()),
        embeddings.process(data['text'].values.tolist()),
    ]
    y = data['category'].values

    classifier.fit(X, y)

    return CategoryModel(embeddings, classifier)


def evaluate(data, category_model):
    X = np.c_[
        category_model.embeddings.process(data['title'].values.tolist()),
        category_model.embeddings.process(data['text'].values.tolist()),
    ]
    y = data['category'].values
    return category_model.classifier.score(X, y)


def predict(title, text, category_model):
    X = np.c_[
        category_model.embeddings.process(title),
        category_model.embeddings.process(text),
    ]
    return category_model.classifier.predict(X)