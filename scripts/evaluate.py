"""
Usage:
    evaluate.py
"""

from upday.data_loader import ModelSaver, DataLoader
from upday import ml

test_data = DataLoader().load(subset='test')
trained_model = ModelSaver().load()

accuracy = ml.evaluate(test_data, trained_model)

print(f'Model accuracy on test data is {100*accuracy:.4f}%')
