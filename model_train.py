import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import hstack
import xgboost
import pickle
from utils import extract_website

def create_training_features(dataset, we_path='website_encoder.pickle'):
    
    def create_website_vector(urls):
        print("getting website vector")
        websites = extract_website(urls)
        encoder = OneHotEncoder(handle_unknown='ignore', drop=None, sparse=True)
        encoder.fit(np.array(websites[~pd.isnull(websites)]).reshape(-1,1))
        pickle.dump(encoder, open(we_path, 'wb'))
        print(f"website encoder saved to {we_path}")
        return encoder.transform(np.array(websites).reshape(-1,1))

    def create_tfidf_vector(corpora, n_features=1000, vec_path = 'tfidf_vectorizer.pickle'):
        print("getting tfidf vector")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.05,
                                        max_features=n_features,
                                        stop_words='english')
        tfidf_vectorizer.fit(corpora)                 
        pickle.dump(tfidf_vectorizer, open(vec_path, 'wb'))
        print(f"tfidf vectorizer saved to {vec_path}")
        return tfidf_vectorizer.transform(corpora)


    def create_label_vector(labels, le_path = 'label_encoder.pickle'):
        print("getting label vector")
        le = LabelEncoder()
        le.fit(labels)
        pickle.dump(le, open(le_path, 'wb'))
        print(f"label encoder saved to {le_path}")
        return le.transform(labels)

    print("loading training data")
    dt = pd.read_csv(dataset, sep='\t')
    print("data loaded")
    X_train_1h = create_website_vector(dt['url'])
    X_train_tfidf = create_tfidf_vector(dt['text'])
    X_train_xg = hstack((X_train_1h, X_train_tfidf))
    y_train_xg = create_label_vector(dt['category'])
    return xgboost.DMatrix(X_train_xg.tocsr(), label=y_train_xg), len(dt['category'].unique())

def train_xgboost(dtrain, num_class, model_path='xgboost.bin'):
    num_round = 300
    print(f"training gradient boosted tree model with {num_round} rounds")
    params = {
        "booster":'gbtree',
        "max_depth": 3,
        "eta": 0.1,
        "objective":"multi:softmax",
        "num_class":num_class
    }
    bst = xgboost.train(params, dtrain, num_round)
    bst.save_model(model_path)
    print(f"xgboost model saved to {model_path}")


if __name__ == "__main__":
    dtrain, num_class = create_training_features('data_redacted.tsv')
    train_xgboost(dtrain, num_class)