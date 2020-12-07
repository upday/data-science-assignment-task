import joblib, sys
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from sklearn import pipeline as skpipe
from sklearn.metrics import classification_report
import pandas as pd

xgb_params = {
            'eta': 0.1, 
            'max_depth': 10,  
            'objective': 'multi:softmax', 
            'verbosity': 3,
            'tree_method': 'auto',
            #'reg_alpha': 3,  # L1
            'silent': False, 
            #'eval_metric':'aucpr',
            'subsample': 0.2,
            'n_jobs' : -1,
            'steps': 25
            } 
            
def train_model(data_source: str):
    
    df = pd.read_csv(data_source, sep="\t", error_bad_lines = False, encoding = "utf-8")
    
    if df.empty:
        return('Aborting')
    
    df_train = df.sample(frac = 0.90)


    df_validation = df.loc[~df.index.isin(df_train.index)]
    vec0 = TfidfVectorizer(strip_accents = 'ascii')
    vec1 = TfidfVectorizer(strip_accents = 'ascii')
    vec2 = CountVectorizer(token_pattern = '[a-zA-Z]{3,15}', strip_accents = 'ascii')
    #A better URL parser will further improve results (Re: beautifulsoup), also not sure if accessing site meta content is permitted, skipping.
    
    mapper = DataFrameMapper([
    ('text', vec0),
    ('title', vec1),
    ('url', vec2)
    ], sparse = True)


    pipeline = skpipe.Pipeline([
         ('featurize', mapper),
         #('mlp', mlp),
         #('rf', RandomForestClassifier(verbose=1)),
         #('lr',LogisticRegression(verbose = 1, class_weight = 'balanced', n_jobs=-1)),
         #('dt',DecisionTreeClassifier())
         ('xgb', XGBClassifier(**xgb_params))
    ],)
    
    print(pipeline[1])
    pipeline.fit(df_train, df_train.category)
    y = pipeline.predict(df_validation)

    print("Report:\n" ,(classification_report(y, df_validation.category)))
    s1 = joblib.dump(pipeline, './models/some_name.pkl')

def test_predictions(data_source: str):
    df_test = pd.read_csv(data_source, sep="\t", error_bad_lines = False, encoding = "utf-8")
    if df_test.empty:
        return('Aborting')
    pipeline = joblib.load('./models/xgboost_all.pkl')
    y = pipeline.predict(df_test)
    print("Report:\n" ,(classification_report(y, df_test.category)))


def predict_unseen(data_source: str):
    df_test = pd.read_csv(data_source, sep="\t", error_bad_lines = False, encoding = "utf-8")
    if df_test.empty:
        return('Aborting')
    pipeline = joblib.load('./models/xgboost_all.pkl')
    df_test['predicted_category'] = pipeline.predict(df_test)
    df_test.to_csv(data_source, sep="\t", mode="w", index=False)
    print("Predictions written to the file")
    return


if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2])