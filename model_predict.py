import argparse
import numpy as np
import pandas as pd
import model_evaluate as me
from scipy.sparse import hstack
import xgboost
import pickle

def get_categories(labels, encoder_path='label_encoder.pickle'):
    le=pickle.load(open(encoder_path, 'rb'))
    return le.inverse_transform(labels.astype(int))

def predict(dataset):
    print(f"loading prediction data: {dataset}")
    dt = pd.read_csv(dataset, sep='\t')
    print(f"data loaded: {len(dt)} rows")
    X_test_1h = me.get_website_vector(dt['url'])
    X_test_tfidf = me.get_tfidf_vector(dt['text'])
    X_test_xg = hstack((X_test_1h, X_test_tfidf))
    dval = xgboost.DMatrix(X_test_xg.tocsr())
    bst = me.get_xgboost()
    pred_labels = bst.predict(dval)
    print(type(pred_labels))
    pred_cat=get_categories(pred_labels)
    for i in range(len(pred_cat)):
        print(f"title: {dt.iat[i, 0]} - predicted category: {pred_cat[i]}")
    
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",help="file with prediction dataset")
    args = parser.parse_args()
    predict(args.dataset)