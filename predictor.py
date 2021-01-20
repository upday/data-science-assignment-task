#######################################################################
# PREDICTOR
#
# This  predictor needs to load a word vector into memory
# As such it should really be listening as a micro-service.
# I will use a restricted word vec here to cut down load time.
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


from funs import getSentenceMeanWv
from funs import joinWordsFromURL

### ****** Input Optparse
optionParser = OptionParser()

usage = "usage: predictor.py [options] arg1 arg2"

optionParser.add_option("-f", "--file_loc", type="string", help="location of file or url as per training data with zip file", default="data/pseudo_test_data.tsv")

optionParser.add_option("-o", "--output_file_loc", type="string", help="location of output file, always writes local", default="data/pseudo_test_data_result.tsv")

# One aspect of the robustness of embedding based models is that they can predict for unseen words - never seen in training. e.g. if 'great' is seen in training and 'wonderful' is not seen, then no matter - the embedding for 'great' is close (e.g. high cosine similarity) to the vector for 'wonderful' - thus the aggregation of one is like the other. 
# So it's best to use the full model - then the model should wait listening really.
optionParser.add_option("-w", "--full_word_vector", type="int", 
        help="use the full word vector or the one restricted to training cases (quicker to load) [0/1]", default =0)

optionParser.add_option("-u", "--url", type="int",
                          help="is file_loc a url or a local file?[0/1]", default = 0)

optionParser.add_option("-l", "--language_detect", type="int",
                          help="perform language detect, otherwise assume English.  If you select 1 this subsequent action is undefined with googletrans down for me right now :) ", default = 0)

# this just assumes a similar structure  to the training data for now..
optionParser.add_option("-z", "--zipped_file_name", type="string",
                          help=" the name of the zipped file - I assume that a url file has a certain structure and name and just use the same as the example", default = "data_redacted.tsv")

# This optparser is not very robust, but it does the trick.

options, arguments = optionParser.parse_args()

verbose = True

if (options.file_loc):
    dataLoc = options.file_loc 
else:
    print( "default missing --file_loc, mysterious, debug")
    sys.exit(1) 

if(options.output_file_loc):
    optLoc = options.output_file_loc
else:
    print("default missing --output_file_loc, mysterious, debug")
    sys.exit(1) 

if(options.full_word_vector is not None):
    useFullWordVec = options.full_word_vector
    if (useFullWordVec != 1):
        useFullWordVec = 0 #it's no unless it's yes
else:
    print("default missing --full_word_vector, mysterious, debug")
    print(options.full_word_vector)
    sys.exit(1) 

if (options.url is not None):
    isUrl = options.url 
    if (isUrl != 1):
        isUrl = 0 #it's no unless it's yes
else:
    print("default missing --url, mysterious, debug")
    sys.exit(1) 

if (options.language_detect is not None): #placeholder code
    doLangDetect = options.language_detect
    if (doLangDetect == 1):
        print("language detection not implemented yet - TODO: figure out translation first")
        sys.exit()


if (options.zipped_file_name):
    zippedFileName = options.zipped_file_name 
else:
    print("default missing zipped_file_name, mysterious, debug")
    sys.exit()

### ***** Functions (I just add them here, since there are not many.


def getSentenceMeanWv(sparseCounts, featureNames, wv, dim = 100, verbose = False):
    # One of my favourite functions... csr_matrix().nonzero()
    # In practice a great complexity reducer for much word count data etc that is highly sparse.
    # This gives a nice way to go through all cases 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.nonzero.html
    # Returns a tuple of arrays (row,col) containing the indices of the non-zero elements of the matrix.
    nonZeroCounts = csr_matrix(sparseCounts).nonzero()

    sentVecs = []
    currCount = 0 #count words per row..
    currSentVec = np.zeros(dim,)
    currRowIdx = 0

    for locIdx in range(len(nonZeroCounts[0])):

        if(verbose and (locIdx % 10000 == 0)):
            print ("{} of {}".format(locIdx, len(nonZeroCounts[0])))

        rowIdx = nonZeroCounts[0][locIdx]
        colIdx = nonZeroCounts[1][locIdx]
        count = sparseCounts[rowIdx,colIdx]
        word = featureNames[colIdx]
        if (word in wv.vocab):
            currWordVec = wv[word]
        else: #funny way of doing it but works... 
            currWordVec = np.zeros(dim,)
            count = 0
        if (rowIdx == currRowIdx):
            currCount += count
            currSentVec += (currWordVec * count) #sum all occurrences of word
        else:
            currRowIdx += 1
            # edge case: missing row in countVector - e.g. all zero counts...
            # above the minum is 29 but let's do it anyway for principle
            while currRowIdx < rowIdx:
                sentVecs.append(np.zeros(dim,))
                currRowIdx += 1

            appendVec = currSentVec / currCount # average it.. e..g tfidf might be better - this is just the general idea..
            sentVecs.append(appendVec) #oof don't ever forget to copy.. ;)
            currSentVec = currWordVec * count #restart vec
            currCount = count
        continue 
    appendVec = currSentVec / currCount
    sentVecs.append(appendVec) #last vec..
    resVec = np.vstack(sentVecs)
    return resVec
    
# small helper for splitting the URL quickly
# could do more here but this is robust enough for the given approach.
def joinWordsFromURL(url):
    res = re.compile(r'[\:/?=\-&]+',re.UNICODE).split(url)
    if (len(res) >= 1):
        return " ".join(res)
    else: return ""



### ****** Setup

# ***Constants

pwd = 'uY4RYREZWSsVZ5Fnhcjgbr' # for this toy I assume the constant password

# *** File Paths
# dataLoc = 'https://upday-data-assignment.s3-eu-west-1.amazonaws.com/science/data.zip' 

dataPrefix = 'data'

modelLoc = os.path.join(dataPrefix, 'trained_rf_model.pkl')
modelVocabLoc = os.path.join(dataPrefix, 'model_vocab.pkl')

fullWordVecFileName = os.path.join(dataPrefix, "enwiki_20180420_100d.txt")

# In prediction I allow a shortcut to load a restricted word vector, to only the words in the training set. But note, this is an unneccessary restriction. A testing case could have many words SEMANTICALLY LIKE the words in training and we would get a similar vector after aggregation. So bear in mind the restriction is more for practical loading. 
# It's worth experimenting with the two on hold-out set. I simplified the cross-validation not to restrict on hod out (partly because my subset function went wrong and kept reference, to fix.. :(
# It's quite easy to have the model listening on a server on an ec2 with the larger model loaded into memory, we used that approach for years at So1 and it worked just fine, for our user load anyway...
restrictedWordVecFileName = os.path.join(dataPrefix, "restricted_enwiki_20180420_100d.txt" )


# *** Read Data

if (isUrl == 1):
    req = urlopen(dataLoc).read()
    file = ZipFile(BytesIO(req))
    dataFile = file.open(zippedFileName, pwd = pwd.encode()) 
    testData = pd.read_csv(dataFile, sep = '\t')
else:
    testData = pd.read_csv(dataLoc, sep = '\t')

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

testData['category'].to_csv(optLoc, sep='\t')

print("\nRESULTS:\n")
print(testData['category'])

sys.exit(0)
#testData = pd.read_csv('data/pseudo_test_data.tsv', sep = '\t')
#testData = testData[['title', 'text', 'url']]
#testData.reset_index(inplace=True)
#testData.columns
#testData.to_json("data/pseudo_test_data.json", orient='records')

# https://towardsdatascience.com/docker-made-easy-for-data-scientists-b32efbc23165
