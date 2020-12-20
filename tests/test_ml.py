import pytest
from upday import ml


def test_embedding_returns_correct_shape():
    embed = ml.TextEmbedding()

    embedded = embed.process(['the long dog jumped over the fence'])

    assert embedded.shape == (1, 50)


def test_embedding_two_strings():
    embed = ml.TextEmbedding()

    embedded = embed.process(['the long dog jumped over the fence',
                              'the poor cat slept under the carpet'])

    assert embedded.shape == (2, 50)
