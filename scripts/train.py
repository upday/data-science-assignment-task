"""
Usage:
    train.py
"""

from upday.data_loader import DataLoader, ModelSaver
from upday import ml

train_data = DataLoader().load('train')
trained_model = ml.train(train_data)
ModelSaver().store(trained_model)
