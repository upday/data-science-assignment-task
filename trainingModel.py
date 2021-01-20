##################################################################
#
# Training Model
# See edaFeatures.py
# I train random forest,
# I use sentence embeddings on the tokenized words from concatenating title, text and url
# That's it.
# I frequently comment on steps.
# See CONCLUSION of edaFeatures.py for a detailed discussion of the decision for this model.
# Note, this only runs locally, not dockerized, I only automate training at a later stage when I understand the problem drift etc. Automating the predictor is more important, obvs.
##################################################################

###
### ****** Imports

### eng

import sys
import os
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen 
#from copy import deepcopy
import random 
import pickle
import re

### ds

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from googletrans import Translator
from langdetect import detect
from scipy.sparse import csr_matrix
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from copy import deepcopy

os.getcwd()

#from upday_data_task.funs import getSentenceMeanWv
#from upday_data_task.funs import joinWordsFromURL

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


# Used for final restriction of word vector to what is used in training
# necessarly to port the CountVectorizer() shortcut and it cuts down on load times if the model is not sitting listening.
def restrictW2v(w2v, restrictedWordSet):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    #new_vectors_norm = []
 
    for i in range(len(w2v.vocab)):
        if (i % 10000 == 0):
            print(i)
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
         #vec_norm = w2v.vectors_norm[i]
        if word in restrictedWordSet:
             #print(i)
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
             #new_vectors_norm.append(vec_norm)
    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)


def subsetW2v(w2v, subsetWordSet, dim = 100, verbose = False):
    newVectors = []
    newVocab = {}
    newIndex2Entity = []
    newIdx = 0

    for i in range(len(subsetWordSet)):
        if (verbose and (i % 10 == 0)):
            print(i)
        word = subsetWordSet[i]
        if word in w2v.vocab:
            vec = w2v[word]
            vocab = w2v.vocab[word]
            vocab.index = newIdx
            newIndex2Entity.append(word)
            newVocab[word] = vocab
            newVectors.append(vec)
            newIdx += 1

    #ahh deep copy, this cause me a WOLRD of pain since it was not a deep copy :( 
    # very strange things hapoen if this function is run repeatedly...
    subsettedWv = deepcopy(w2v)
    #subsettedWv = Word2Vec(size = dim)
    subsettedWv.vocab = newVocab
    subsettedWv.vectors = np.array(newVectors)
    subsettedWv.index2entity = np.array(newIndex2Entity)
    subsettedWv.index2word = np.array(newIndex2Entity)
    return subsettedWv


### ****** Setup


# *** Constants

# just a logic argument for whether to use the language logic as is or drop for new data - that would require proper translation approach or embedding alternative (the latter would require some change of code also re CountVectorizer().. )
trainOnNewData = True
doRestrictWordVec = False

pwd = 'uY4RYREZWSsVZ5Fnhcjgbr' # I would keep this outside any docker image for real prod code and pass as a variable to the docker build but will just code it in here for now... (but I know I really shouldn't!)


# *** File Paths
dataPrefix = 'data'
dataLoc = 'https://upday-data-assignment.s3-eu-west-1.amazonaws.com/science/data.zip' 
modelLoc = os.path.join(dataPrefix, 'trained_rf_model.pkl')
modelVocabLoc = os.path.join(dataPrefix, 'model_vocab.pkl')

# So googltrans broke my heart a little here. It's becoming a war of attrition.
# I know it's just one case, but it's the idea of tranlsating and then applying a unified approach that I wanted to demonstrate.
# An alternative is to try to use wordVectors from different langauge, as per fasttext:aligned vectors - not any ones can do - they need to be aligned.
# See here: https://fasttext.cc/docs/en/aligned-vectors.html
# There are various approaches to that such as CCA https://www.aclweb.org/anthology/E14-1049.pdf
# But in practice I think (in limited experience when I tested before) translation and embedding is safer than embedding then rotation, and there is less model overhead etc..
huDataBkup = pd.read_csv(os.path.join(dataPrefix, 'upday_hungarian_data.tsv'), sep='\t')

# I wasted a lot of time trying to make subsets of this. Disappointed in myself and want to reskill to get back to speed here. It's good to be able to open up and manipulate these models.
fullWordVecFileName = os.path.join(dataPrefix, "enwiki_20180420_100d.txt")
restrictedWordVecFileName = os.path.join(dataPrefix, "restricted_enwiki_20180420_100d.txt" )


# *** Read Data

# https://stackoverflow.com/questions/31967635/read-zipfile-from-url-into-stringio-and-parse-with-panda-read-csv
req = urlopen(dataLoc).read()
file = ZipFile(BytesIO(req))
dataFile = file.open("data_redacted.tsv", pwd = pwd.encode())
data = pd.read_csv(dataFile, sep = '\t')

print("reading  word vector model") #fyi...
wordVec = KeyedVectors.load_word2vec_format(fullWordVecFileName, binary=False)
print("word vector model loaded")


### *** Data pre-processing

# *** Language
# Note, this is just a placeholder for more the kind of idea. 
# I assume the single case might represent a growing number of non-English articles
# In this version train/test, I just deal with Hungarian, and use the backup tsv when googletrans fails.  The version of googletrans I am using was working in December. it's a game of cat and mouse. 2 or 3 years ago I was in favour of hacking translation, but it it's business important I think it's worht paying fow now - theere are other options, e.g. Amazon.
# Note NBNB, if business-wise translation is too expensive, and too unreliable when hacked, consider trying fasttext aligned word vectors, i.e. just detect the language, use the appropirate aligned embedding, and then apply similar techniques... 
# That would not work with my CountVectorizer() approach, but that's just one approach I find quite efficient on my local machine (because the non-sparse cases are in practice and order ot two less than the dimensionality of the counts matrix (can't formalize that off the top of my head) . You can always use pyspark etc to distribute a slower line by line algorthim...

data['lang'] = data['text'].apply(detect)

# this is really just to illustrate the general approach..
# i.e. translate then embed..
# So if it's new data I will just skip translating for now, since I need to figure out a solution to the issue, e.g. vpns, actually paying, new fix to googletrans, etc...
if (trainOnNewData is False):

    huData = data.loc[data['lang'] == 'hu']
    translator = Translator()
    try:
        huDataTitleTrans = huData['title'].apply(translator.translate, src='hu', dest='en').apply(getattr, args = ('text',))
        huDataTextTrans = huData['text'].apply(translator.translate, src='hu', dest='en').apply(getattr, args = ('text',))
        #huDataTextTrans = huDataTitleTrans
        huDataTransList = [huDataTitleTrans, huDataTextTrans, huData['url'], huData['category'] ]
        huDataTrans = pd.concat(huDataTransList, keys = huData.columns, axis = 1)
    except:
        huDataTrans = huDataBkup
    
    huDataTrans['lang'] = 'hu'
    data = pd.concat([ data[data['lang']=='en'], huDataTrans],
        axis = 0)
else:
    print("dropping non-English cases")
    data = data.loc[data['lang'] == 'en']
    if (data.shape[0] == 0):
        print ("no non-English cases, no model, bye")




# *** text combination

data['url_words'] = data['url'].apply(joinWordsFromURL)
data['text_title'] = data['title'] + ' ' + data['text']
data['text_title_url'] = data['title'] + ' ' + data['text'] + ' ' + data['url_words']


### *** Feature Creation

# *** CountsVectorizer(), implicit tokenization, cleaning etc...

ttuNoStopCountVectorizer = CountVectorizer(stop_words='english')
ttuNoStopCounts = ttuNoStopCountVectorizer.fit_transform(data['text_title_url'])
ttuNoStopWords = ttuNoStopCountVectorizer.get_feature_names()

# *** Sentence Emeddings Aggregation


ttuNoStopSentenceVecs = getSentenceMeanWv(ttuNoStopCounts, ttuNoStopWords, wordVec, verbose = True)

# and that's all...


### ******* Train Model

# I'm deliberately using a very simple model here. It's so robust on these kind of features and I did not get a huge gain from (random) searching the parameter space.
# Same with XGG defaults, only a small improvement.
# I think if we had more empirical features (semi-rules based) then we might get more of those kind of if-else cuts in the decision trees, that might be more sensitive to ensemble learner parameters.
# But for an MVP I think simplicity is strength sometimes.


ttuNoStopRf = RandomForestClassifier()
ttuNoStopRf.fit(ttuNoStopSentenceVecs, data['category'])


### ^^ trained
with open(modelLoc, 'wb') as modelFile:
    pickle.dump(ttuNoStopRf, modelFile)

# Save the vocab for the CountsVectorizer() in test
with open(modelVocabLoc, 'wb') as f:
    pickle.dump(ttuNoStopWords, f)

# Finally restrict the word vector so that we only need to load one that matches the training dimensions... new words will be thrown away... this is why we also save the modelVocab
#  

if (doRestrictWordVec):
    #print ("restricting word vec, this is very slow")
    # this is very very slow...
    #restrictW2v(wordVec, ttuNoStopWords)
    # See funs.py ^^^

    # So I made this, but it has nasty reference bug in it..
    # So I can only run it once at the end (caused me a world of pain when I was comparing several in my preparatory code :
    # 
    print("subsetting word vec - quicker")# a copy bug there so use wisely..

    subsettedWv = subsetW2v(wordVec, ttuNoStopWords, verbose = True)

    subsettedWv.save_word2vec_format(restrictedWordVecFileName, binary=False)
