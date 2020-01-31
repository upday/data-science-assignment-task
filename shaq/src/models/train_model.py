import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score


def train_test_data():
    """transform cleaned text data into features and labels, and split into training and testing data"""
    sample_data = pd.read_pickle(os.getcwd() + '/data/processed/sample_dat_1.pkl')  # read preprocessed data

    # transform cleaned text data to features
    feature_transformer = TfidfVectorizer(sublinear_tf=True, norm='l2')
    features = feature_transformer.fit_transform(sample_data.cleaned_text)
    joblib.dump(feature_transformer, os.getcwd() + '/models/transformer.pkl') # save transformer to encode new data

    target_data = sample_data['category_id']

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        target_data,
                                                        test_size=0.3,
                                                        stratify=target_data,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """train selected model and save"""
    lr_model = LogisticRegression(solver='lbfgs', multi_class='auto')
    lr_model.fit(X_train, y_train)

    joblib.dump(lr_model, os.getcwd()+'/models/lr_model.pkl')
    print("-" * 50)
    print("model uploaded")
    print("-" * 50)

    return lr_model

def eval_model(X_test, y_test):
    """evaluated selected model on test data"""
    model = joblib.load(os.getcwd()+'/models/lr_model.pkl')
    y_pred = model.predict(X_test)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7) # for sake of speed reduced number of splits to 3

    print("-"*50)
    print('Accuracy (Test Data)', round(model.score(X_test, y_test), 2))
    print('Accuracy (Training Data)', round(cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy').mean(), 2))
    print('F1-Score (Test Data)', round(f1_score(y_test, y_pred, average='weighted'), 2))
    print('F1-Score (Training Data)', round(cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_weighted').mean(), 2))
    print('Precision (Test Data)', round(precision_score(y_test, y_pred, average='weighted'), 2))
    print('Precision (Training Data)', round(cross_val_score(model, X_train, y_train, cv=kfold, scoring='precision_weighted').mean(), 2))
    print('Recall (Test Data)', round(recall_score(y_test, y_pred, average='weighted'), 2))
    print('Recall (Training Data)', round(cross_val_score(model, X_train, y_train, cv=kfold, scoring='recall_weighted').mean(), 2))
    print("-" * 50)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_test_data()
    train_model(X_train, y_train)
    eval_model(X_test, y_test)


