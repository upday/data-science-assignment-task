from copy import deepcopy
import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


# Function to get the mean sentence vector for all training cases
# I use the CountVectorizer sparse matrix - this seems efficient because:
# 1. It takes care oof tokenization internally - happy with this for a quick algorithm
# 2. I then use the csr_matrix().nonzero() array tuple to quickly iterate
# Creating test cases is not as trivial though


# trainFeatureNames the get_feature_names from the train CountVectorizer
# testCases the testCases as a python series ['text'] +['title']
# This is a bit of a hack - I tokenize testCases, remove from 
#def getTestSentenceMeanWv(trainFeatureNames, testCases, dim = 100, verbose = False):

def getSentenceMeanWv(sparseCounts, featureNames, wv, dim = 100, verbose = False):

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

    

def joinWordsFromURL(url):
    res = re.compile(r'[\:/?=\-&]+',re.UNICODE).split(url)
    if (len(res) >= 1):
        return " ".join(res)
    else: return ""

def getPaperFromURL(url):
    res = re.compile(r'[\:/?=\-&]+',re.UNICODE).split(url)
    if (len(res) >= 2):
        return res[1]
    else:
        return "NOTFOUND"


# https://stackoverflow.com/questions/50914729/gensim-word2vec-select-minor-set-of-word-vectors-from-pretrained-model
# https://stackoverflow.com/questions/50914729/gensim-word2vec-select-minor-set-of-word-vectors-from-pretrained-model
#def restrictW2v(w2v, restrictedWordSet):
#    new_vectors = []
#    new_vocab = {}
#    new_index2entity = []
#    #new_vectors_norm = []
#
#    for i in range(len(w2v.vocab)):
#        if (i % 10000 == 0):
#            print(i)
#        #print(i)
#        word = w2v.index2entity[i]
#        vec = w2v.vectors[i]
#        vocab = w2v.vocab[word]
#        #vec_norm = w2v.vectors_norm[i]
#        if word in restrictedWordSet:
#            #print(i)
#            vocab.index = len(new_index2entity)
#            new_index2entity.append(word)
#            new_vocab[word] = vocab
#            new_vectors.append(vec)
#            #new_vectors_norm.append(vec_norm)
#    w2v.vocab = new_vocab
#    w2v.vectors = np.array(new_vectors)
#    w2v.index2entity = np.array(new_index2entity)
#    w2v.index2word = np.array(new_index2entity)
#    #w2v.vectors_norm = np.array(new_vectors_norm)



# based on https://stackoverflow.com/questions/48941648/how-to-remove-a-word-completely-from-a-word2vec-model-in-gensim
# https://stackoverflow.com/questions/50914729/gensim-word2vec-select-minor-set-of-word-vectors-from-pretrained-model
# A lot quicker when the restricted vocab size is much less than the word vec vocab size
# Need to debug this
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

