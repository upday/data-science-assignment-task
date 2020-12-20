import pytest
from upday import data_loader

def test_data_loader_loads_the_same_training_set_every_time():
    dl = data_loader.DataLoader()
    train_set1 = dl.load('train')
    train_set2 = dl.load('train')
    assert all(train_set1 == train_set2)