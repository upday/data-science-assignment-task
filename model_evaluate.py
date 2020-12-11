import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score,f1_score
from scipy.sparse import hstack
import xgboost
import pickle
import argparse
from utils import extract_website

def get_website_vector(urls, encoder_path='website_encoder.pickle'):
    encoder=pickle.load(open(encoder_path, 'rb'))
    websites = extract_website(urls)
    return encoder.transform(np.array(websites).reshape(-1,1))

def get_tfidf_vector(corpora, vectorizer_path='tfidf_vectorizer.pickle'):
    tfidf_vectorizer=pickle.load(open(vectorizer_path, 'rb'))
    return tfidf_vectorizer.transform(corpora)

def get_label_vector(labels, encoder_path='label_encoder.pickle'):
    le=pickle.load(open(encoder_path, 'rb'))
    return le.transform(labels)

def get_xgboost(model_path='xgboost.bin'):
    bst = xgboost.Booster({'nthread': 4}) 
    bst.load_model(model_path)
    return bst

def evaluate_model(dataset):
    print(f"loading evaluation data: {dataset}")
    dt = pd.read_csv(dataset, sep='\t')
    print("data loaded")
    X_eval_1h = get_website_vector(dt['url'])
    X_eval_tfidf = get_tfidf_vector(dt['text'])
    X_eval_xg = hstack((X_eval_1h, X_eval_tfidf))
    y_eval_xg = get_label_vector(dt['category'])
    dval = xgboost.DMatrix(X_eval_xg.tocsr(), label=y_eval_xg)
    bst = get_xgboost()
    pred = bst.predict(dval)
    acc = accuracy_score(y_eval_xg, pred)
    f1=f1_score(y_eval_xg, pred, average='weighted')
    print(f"model accuracy: {acc}")
    print(f"model weighted f1 score: {f1}")
    
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",help="file with validation dataset")
    args = parser.parse_args()
    evaluate_model(args.dataset)
