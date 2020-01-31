import os
import sys
import joblib
import numpy as np


def predict(new_data):
    """"Predicts article category"""
    # load model
    loaded_model = joblib.load(os.getcwd()+'/models/lr_model.pkl')

    # load transformer
    loaded_transformer = joblib.load(os.getcwd()+'/models/transformer.pkl')

    # make prediction
    test_features = loaded_transformer.transform([new_data])
    y_pred = loaded_model.predict(test_features)

    # category mapping, for now fixed in the script
    target_mapping ={0: 'technology_science',
                     1: 'digital_life',
                     2: 'sports',
                     3: 'fashion_beauty_lifestyle',
                     4: 'money_business',
                     5: 'politics',
                     6: 'people_shows',
                     7: 'news',
                     8: 'culture',
                     9: 'travel',
                     10: 'music',
                     11: 'cars_motors'}

    return print("Predicted category for this article:", np.vectorize(target_mapping.get)(y_pred))

if __name__ == '__main__':
    try:
        predict(sys.argv[1])
    except:
        print("Script Error: Category can not be predicted. Please pass a test article and run script again.")