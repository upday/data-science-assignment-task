from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrices(X_test):
    heatmap_data = X_test.groupby(['category', 'predicted_category']).apply(len).reset_index().pivot('category', 'predicted_category', 0)
    fig, axes = plt.subplots(figsize=(20, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap="YlGnBu")
    plt.title('Confusion Matrix');

    
    column_sums = heatmap_data.sum(axis=0)
    precision_heatmap_data = heatmap_data / np.array(column_sums)[np.newaxis, :]
    fig, axes = plt.subplots(figsize=(20, 10))
    sns.heatmap(precision_heatmap_data, annot=True, cmap="YlGnBu", fmt='.2f')
    plt.title('Precision Matrix');

    row_sums = heatmap_data.sum(axis=1)
    recall_heatmap_data = heatmap_data / np.array(row_sums)[:, np.newaxis]
    fig, axes = plt.subplots(figsize=(20, 10))
    sns.heatmap(recall_heatmap_data, annot=True, cmap="YlGnBu", fmt='.2f')
    plt.title('Recall Matrix');

    
def category_histogram(df):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.countplot(data=df, y='category')
    plt.grid()
    plt.title('Histogram of categories')