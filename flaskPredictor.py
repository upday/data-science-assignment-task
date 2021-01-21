
#######################################################################
# FLASK PREDICTOR
#
# Cogged from here basically: https://towardsdatascience.com/docker-made-easy-for-data-scientists-b32efbc23165
#
# This  predictor reads json from the api
# See data/psuedo_test_data.json
# It's creatable from the tsv via pd.to_json(orient='records'), removing all columns except 'title', 'text', 'url'
#######################################################################

### ****** Imports

### eng

import sys
import os
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen 
import random 
import pickle
import re
import logging

from optparse import OptionParser


### ds

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from flask import Flask, request, jsonify

from funs import getSentenceMeanWv
from funs import joinWordsFromURL

app = Flask(__name__)


a = 10

# *** CONSTANTS - CHANGE THESE TO CHANGE MODEL
useFullWordVec = 0 #for efficiency it does not use this but should perform better with it... toggle this as you like.  

# *** File Paths
    
dataPrefix = 'data'
    
modelLoc = os.path.join(dataPrefix, 'trained_rf_model.pkl')
modelVocabLoc = os.path.join(dataPrefix, 'model_vocab.pkl')
    
fullWordVecFileName = os.path.join(dataPrefix, "enwiki_20180420_100d.txt")
    
# In prediction I allow a shortcut to load a restricted word vector, to only the words in the training set. But note, this is an unneccessary restriction. A testing case could have many words SEMANTICALLY LIKE the words in training and we would get a similar vector after aggregation. So bear in mind the restriction is more for practical loading. 
# It's worth experimenting with the two on hold-out set. I simplified the cross-validation not to restrict on hod out (partly because my subset function went wrong and kept reference, to fix.. :(
# It's quite easy to have the model listening on a server on an ec2 with the larger model loaded into memory, we used that approach for years at So1 and it worked just fine, for our user load anyway...
restrictedWordVecFileName = os.path.join(dataPrefix, "restricted_enwiki_20180420_100d.txt" )
    
    
    # *** Read Data
    
    
with open(modelLoc , 'rb') as file:
    model = pickle.load(file)
    
with open(modelVocabLoc, 'rb') as file:
    modelVocab = pickle.load(file)
    
if (useFullWordVec == 1):
    print("loading full word vector, this takes time")
    wordVec = KeyedVectors.load_word2vec_format(fullWordVecFileName, binary=False)
    print("word vector loaded")
else:
    print("loading restricted word vector, this takes less time but may affect performance since words unseen in training can still be informative")
    wordVec = KeyedVectors.load_word2vec_format(restrictedWordVecFileName, binary=False)
    #subWordVec = KeyedVectors.load_word2vec_format(restrictedWordVecFileName, binary=False)
    print("word vector loaded")


@app.route('/predict', methods=['POST'])

def predict():

    print("start predict")

    #testJsonCases = request.json
    testJsonCases = request.get_json(force=True)

    print("got request json")

    #TODO
    try:
        testData = pd.DataFrame(testJsonCases)
    except:
        # ValueError: If using all scalar values, you must pass an index
        testData = pd.DataFrame(testJsonCases, index=[0])

    logging.warning("testData.columns")
    logging.warning("{}".format(testData.columns))

    logging.warning("testData.shape")
    logging.warning("{}".format(testData.shape))

    logging.warning("testData")
    logging.warning("{}".format(testData))

    logging.warning("input test data")
    logging.warning("{}".format(testData))

    testData.columns = ['title','text', 'url']


    verbose = True
    
    ### ****** Setup
    
    
    ###### *** Data Processing
    
    ## No language stuff for now..
    
    testData['url_words'] = testData['url'].apply(joinWordsFromURL)
    testData['text_title'] = testData['title'] + ' ' + testData['text']
    testData['text_title_url'] = testData['title'] + ' ' + testData['text'] + ' ' + testData['url_words']
    
    
    if (useFullWordVec == 1):
        testCountVectorizer = CountVectorizer(stop_words='english')#hardcode the stopwords as before - just makes sense I think anyway..
    else:
        testCountVectorizer = CountVectorizer(vocabulary=modelVocab, stop_words='english')
    
    testCounts = testCountVectorizer.fit_transform(testData['text_title_url'])
    testWords = testCountVectorizer.get_feature_names()
    
    # *** Sentence Emeddings Aggregation
    
    testSentenceVecs = getSentenceMeanWv(testCounts, testWords, wordVec, verbose = True)
    
    # *** predict and add to testData
    
    res = model.predict(testSentenceVecs)
    testData['category'] = res
    
    # *** output results
    
    return jsonify({'prediction': list(testData['category'])})

    #testData = pd.read_csv('data/pseudo_test_data.tsv', sep = '\t')
    #testData = testData[['title', 'text', 'url']]
    #testData.reset_index(inplace=True)
    #testData.columns
    #testData.to_json("data/pseudo_test_data.json", orient='records')
    
    # https://towardsdatascience.com/docker-made-easy-for-data-scientists-b32efbc23165


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

testData = pd.read_csv('psuedo_test_data.tsv', sep='\t')
