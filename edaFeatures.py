


######################################################################
# EDA, Feature Creation and Model Training for Upday Coding Challenge
######################################################################
## Note on the files:
## I work in a vim-tmux environment, so I write here with an REPL underneath. 
## e.g. as explained here:
## https://towardsdatascience.com/getting-started-with-vim-and-tmux-for-python-707ec5ff747f
## Once you have the vim shortcuts down this leaves Jupyter in the dust.
## One drawback is that it is not possible to print the output of the interpreter back into the text file.
## With the N-vimR plug-in I can do this for R - it makes data evaluation safe, smooth and easy.
## Here it's easy enough to copy output back with tmux shortcuts. It's not too annoying for me because I am generally taking a moment to look at the output anyway.
## I don't see any need to embed plots in this. I could plot some sort of tsne reduction with different colours per category but it would really just be a pretty picture.
## If I need something to present like that, I would copy into a jupyter notebook at the end and present that way.
### A really lovely advantage is how easily it is to set up on an ec2 instance when dealing with larger data sets and for large matrix multiplications etc..
##
## Yes, I use lower camel. I know. :|

# Note, this only runs locally, not dockerized, it's just exploratory code.
##############################################################


### ****** Imports

### eng

import sys
import os
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen 
from copy import deepcopy
import random 
import pickle
import re

### ds

import pandas as pd
import numpy as np

#import spacy
#import nltk  # note to self: some lazy imports here...  
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer #later step.
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
#from textblob import TextBlob
from googletrans import Translator
from langdetect import detect
from scipy.sparse import csr_matrix
#from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import xgboost as xgb

### own
os.getcwd()

# change as required.
packageDir = '/home/simon/Work/code/repos/coding_challenges/upday/upday_data_task'
sys.path.append(os.path.normpath(os.path.join(packageDir, '..')))

from upday_data_task.funs import subsetW2v
from upday_data_task.funs import getSentenceMeanWv
from upday_data_task.funs import getPaperFromURL
from upday_data_task.funs import joinWordsFromURL



# to delte...
os.chdir(packageDir)
# d = pd.read_csv('data/data_redacted.tsv', sep='\t')




### ****** Setup

# *** Options 

pd.set_option('display.max_colwidth', None)


# *** Constants
pwd = 'uY4RYREZWSsVZ5Fnhcjgbr' # I would keep this outside any docker image for real prod code and pass as a variable to the docker build but will just code it in here for now... (but I know I really shouldn't!)


# *** File Paths

dataPrefix ="data"
dataLoc = 'https://upday-data-assignment.s3-eu-west-1.amazonaws.com/science/data.zip' 

huDataBkup = pd.read_csv(os.path.join(dataPrefix, 'upday_hungarian_data.tsv'), sep='\t')

# https://wikipedia2vec.github.io/wikipedia2vec/pretrained/#english
# From https://wikipedia2vec.github.io/wikipedia2vec/intro/
fullWordVecFileName = "enwiki_20180420_100d.txt"


# *** Read Data

# https://stackoverflow.com/questions/31967635/read-zipfile-from-url-into-stringio-and-parse-with-panda-read-csv
req = urlopen(dataLoc).read()
file = ZipFile(BytesIO(req))
dataFile = file.open("data_redacted.tsv", pwd = pwd.encode())
data = pd.read_csv(dataFile, sep = '\t')


### ****** Basic EDA 


data.shape
# (8646, 4)

data.columns
## Index(['title', 'text', 'url', 'category'], dtype='object')


data['title']
# 0                            Don"t Mourn the Y Chromosome
# 1       Destiny: Bungie to introduce ammo pack micro-t...
# 2       Daruvala to race for Josef Kaufmann Racing in ...
# 3       Secondary tropical forests absorb carbon at hi...
# 4       This Brow Hack Will Change the Way You Fill In...
# ...
# 8641            Is it more hygienic to remove pubic hair?     !!!
# 8642    Justin Timberlake and Jessica Biel Are So Swee...
# 8643     Two Cats Kitchen, Birmingham – restaurant review
# 8644    5 Tips From a Health Expert to Finally Get Hea...

data['text']
# 0       The human Y chromosome may vanish one day, but...
# 1       Bungie is putting together plans to start sell...
# 2       Highly rated Jehan Daruvala will race for Jose...
# 3       Forests are an important carbon sink. While mo...
# 4       Before everyone was trying to get eyebrows on ...
# ...
# 8641    Since when did pubic hair become so yucky? Sur...
# 8642    What’s Your Reaction? Thanks for your reaction...
# 8643    ‘What the hell is “new Baltic cuisine” when it...
# 8644    If achieving a healthier lifestyle is one of y...
# 8645    The firm has completed three years as the nort...
# 8645    Business Finance Solutions hits £7.5m mileston...

data['url']
# 0           http://discovermagazine.com/2014/nov/13-y-not
# 1       http://www.ibtimes.co.uk/destiny-bungie-introd...
# 2       http://www.thecheckeredflag.co.uk/2015/12/daru...
# 3       http://www.sciencedaily.com/releases/2016/02/1...
# 4       http://www.popsugar.com/beauty/How-Use-Brow-Ge...
# ...
# 8641    https://www.theguardian.com/lifeandstyle/2016/...
# 8642    http://www.popsugar.com/celebrity/Justin-Timbe...
# 8643    http://www.theguardian.com/lifeandstyle/2015/s...
# 8644    http://www.inc.com/derek-flanzraich/5-tips-fro...
# 8645    http://www.manchestereveningnews.co.uk/busines...

# Note: we can extract some important info here I think....

data['category']
# 0             technology_science
# 1                   digital_life
# 2                         sports
# 3             technology_science
# 4       fashion_beauty_lifestyle
# ...
# 8641    fashion_beauty_lifestyle
# 8642                people_shows
# 8643    fashion_beauty_lifestyle
# 8644    fashion_beauty_lifestyle
# 8645              money_business


# So it's NLP categorization. 
# Certainly, if there were more rows of data I would be very optimistic for something like 
# the supervised version of the fast_text algorithm.  
# I am not so hopeful that we have enough rows here to learn the word embeddings.
# It's bag of words so we could always also reduce the dimensionality by removing stop words etc..
# I will do some diagnostic checks of data statistics anyway....

data.isnull().sum()
# title       0
# text        0
# url         0
# category    0


### ****** Class Variable Checks

# *** category
## class distributions

data['category'].value_counts()
# fashion_beauty_lifestyle    1398
# sports                      1165
# technology_science          1040
# digital_life                 738
# money_business               733
# news                         646
# music                        568
# culture                      547
# travel                       544
# cars_motors                  470
# politics                     430
# people_shows                 367

numCats = len(data['category'].unique())
## 12 categories, that's quite a lot for the number of rows..


## Cf. https://www.datascienceblog.net/post/machine-learning/performance-measures-multi-class-problems/
## Performance Measures for Multi-Class Problems - Data Science Blog: Understand. Implement. Succed.

## Consider multiclass log loss --- https://datascience.stackexchange.com/questions/31315/good-performance-metrics-for-multiclass-classification-problem-besides-accuracy


## "The solution should just perform better than random.."
## What should random mean here?

## Let's make two random baselines, not sure that we'll use them....

def dumbRandom(predData):
    return random.choices(
            predData['category'].unique(), 
            weights = [1] * predData['category'].nunique(), 
            k = predData.shape[0])

## For fair 12 sided dice we would have...
def dumbRandomProb(numCats = 12):
    return 1 / numCats # pick at random

def dumberRandom(predData):
    return [(predData['category'].mode().tolist()[0])] * predData.shape[0] #any mode will do

def dumberRandomProb():
    return data['category'].value_counts().max() / data.shape[0] # pick the most frequent

dumbRandom(data[:5]) # is random
# e.g. ['digital_life', 'digital_life', 'technology_science', 'sports', 'technology_science']


dumbRandomProb()
# 0.08333333333333333

metrics.accuracy_score(
        y_true = data['category'],
        y_pred = dumbRandom(data)) 
# 0.08431644691186677

sum(
    [metrics.accuracy_score(
        y_true = data['category'],
        y_pred = dumbRandom(data)) for i in range(1000)]) / 1000
# 0.08348461716400643

# etc...


# close enough...

dumberRandom(data[:5])
# i.e. ['technology_science', 'technology_science', 'technology_science', 'technology_science', 'technology_science']

dumberRandomProb()
# 0.16169326856349758

metrics.accuracy_score(
        y_true = data['category'], 
        y_pred = dumberRandom(data))
# 0.16169326856349758

## I love that sometimes dumber random beats dumb random 


# We assume our training data is representative of the true distribution - 
# If not we address that at the data collection stage or weight training cases later based on the observed true distribution..


### ****** Metric Functions
# I can see no reason why one mistake is worse than any other, based on my current understanding of the data. 
# I don't see why precision/recall for any class has greated business importance at this stage. This is the kind of thing to review over time - make sure the metric matches the business problem and goals.
# So I could use Accuracy and try to beat the dumberRandomProb() baseline. 
# Another approach is to use the Cohen's Kappa score, which measures how much an estimator is better than chance. A kappa greater than 0 is better than chance. 
# Kappa = (observed accuracy - expected accuracy)/(1 - expected accuracy)
# https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
# In practice, I will track both of these, also as a kind of sanity check. 
# I could also work out chance Micro/Macro F1-Scores and compare, but I think just tracking Accuracy and Kappa will suffice.
# 


### *** URL

data['url']



data['paper'] = data['url'].apply(getPaperFromURL)

data['paper'].value_counts()
# www.theguardian.com          903
# www.telegraph.co.uk          440
# www.independent.co.uk        431
# www.bbc.com                  240
# www.dailymail.co.uk          208
#                             ...


data['paper'].value_counts().shape
# (1359,)
# 1359 papers - I could pick a few but many use all of these categories - maybe not so much science in the DailyMail, lol, but I won't go into that here...


# I will do this later after the translation case...
#data['url_words'] = data['url'].apply(lambda u: ' '.join(getWordsFromURL))
#data['url_words']
## 8645            http www.manchestereveningnews.co.uk business business news business finance solutions hits 75m 10423182
data['category']
# 8645              money_business

# I'll just throw it in raw, the newspapers will be missed by the wordvector as desired...
# I will sum all words, including repeats and average the sentence embeddings
# So I like these repeats of business here. :) 
# That is, the URL words will just double up the semantic signal...




# I am going to skip the URL because it primarily provides some rule based additions
# o


### ****** Models
# With more data I might try something hispec like the fasttext supervised learner.
# Here I will probably just use a MultinomialNB. In real life I would compare several models e.g. an SVM.
# I am not so worried about that here for the following reason:
# Comparing lots of different models would , imo, kind of give a false impression of due diligence.
# I will need to do a large amount of text processing here, and I am not going to have (or make if you prefer) time to properly grid search the different steps I can take here, e.g. fully check the impact of removing/ not removing stop-words etc.
# That is, I will be doing some data-processing that 'seems legit' after EDA and making a stab at a model based on that. 
# For a through search over feature creation methods I would structure the code more with feature pipelines etc to allow for a more organized approach. 
# So please consider this just a first stab at the problem.

###



# https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1


# https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c

# WHICH METRIC FUNCTION TO USE???

# WHich algorithm? -> consider 
# SVM...
# MultinomialNB
# https://www.kaggle.com/getting-started/42409 <---------------------
# tf/idf for NB <-- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# w2v features??? <------- pre-trained <---- TODO
# NOT Bert vectors for now...

# NBNB https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a

# accuracy (micro f1 score)
# weighted accuracy
# macro f1 score
# weighted f1 score
# kappa?


### ****** Metric Functions


# For the purposes of this task, the confusion_matrix and classification_report are not really surveyable.
# I don't know the meanings of the different classes. 
# I'm inclined to think that assuming all errors are equal is a fair enough first estimate.

# metrics.confusion_matrix(dumbRandom(data), data['category'])
# metrics.classification_report(dumbRandom(data), data['category'])


### ****** Independent Variable Checks


# I will look at these separately for now, but...
# NB: Due to the low number of rows it may make sense to combine the columns at least for some features..

# (Independet in the sense that they are not y, the dependent variable, but since we use Naive Bayes the name kind of sticks)

titleCountVectorizer = CountVectorizer(stop_words = 'english') #It's not semantic per se so I will remove these for dimenstionality etc.. but for prod I would compare both approaches...

### *** Title

titleCounts = titleCountVectorizer.fit_transform(data['title'])

titleCountsArr = titleCounts.toarray()

print(titleCountsArr.shape)
# (8646, 15875)

titleWordCounts = titleCountsArr.sum(axis = 1)
titleUniqueWordCounts = (titleCountsArr != 0).sum(axis = 1)

pd.Series(titleWordCounts).describe()
# count    8646.000000
# mean        6.622831
# std         2.754859
# min         0.000000
# 25%         5.000000
# 50%         6.000000
# 75%         8.000000
# max        36.000000

pd.Series(titleUniqueWordCounts).describe()
# count    8646.000000
# mean        6.569396
# std         2.675205
# min         0.000000
# 25%         5.000000
# 50%         6.000000
# 75%         8.000000
# max        35.000000



# A lot of words per rows.
# I really want to just drastically reduce the dimensionality for this task...
# hang on though...


titleWords = titleCountVectorizer.get_feature_names()
# ...'érdeklődés', 'és', 'öntanuló', 'önvezető', 'östersund', 'ötzi']
# we got some funny looking words there

weirdLangCase = data[data['title'].str.contains('öntanuló')]
print(weirdLangCase['url'])
# 6925    https://www.chameleon-smarthome.com/sajto/mti-hirek-2019

print(weirdLangCase['text'])
# ... százalékra emelkedik 2021-re. Európában a smart home beruházások összértéke évente átlagosan 39 százalékkal emelkedett az elmúlt időszakban, míg 2020-ra az európai piac mérete várhatóan meghaladja a 23 milliárd eurót. A cég tervei között szerepel, hogy két éven belül piacra lépnek az Egyesült Államokban, mely globális szinten jelenleg a legnagyobb smart home piac, 24 milliárd euró, azaz közel 8,000 milliárd forint értékkel. A magyar startup idén 250-300 millió forint árbevételt remél, a tavalyi közel 50 millió forint után.

# Hungarian...

### *** Digression 1: Language Checks
# I will run a lang i.d. on the text fields for each to try to detect languages
# I use langdetect because it seems not to hit the google api.
# A pro solution would use a more formal approach (depending on budget...)
# I demonstrate googletrans here anyway.. I mostly just want to show awareness of the issue..
# It used to be so much easier, alas!


data['lang'] = data['text'].apply(detect)

data['lang'].value_counts()
# en    8645
# hu       1

# lol, 1 rogue Hungarian case.. 
# googletrans is a world of pain these days, it's so hard to write automated code without a licence anymore
# here I am wrapping something that "SHOULD WORK" in a try/except that defaults to 'one I prepared earlier'


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

### *** end Digression 1
### *** Title contd...

# I am going to try different approaches for different things...
# For Title, I will include it with Text for the Text feature - I think - why not?
# So let's do something different with it on its own too...
# Maybe just an aggregation of an embedding.


## >>>> WORD EMBEDDING CODE HERE
# https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92

# https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

# I love training word embeddings, but for this size of data I think pretrained makes sense.
# I deliberately avaoided anything trained on WSJ corpus, going for the more general wikipedia.
# In general I would certainly consider something more context specific, but I don't know how representative WSJ is of all these categories
# Basically, I don't know about using embeddings from 1 newspaper for a range of papers..
# I use word2vec first because I understand the algorithm better than the others.
# But I would compare several different embeddings, e.g. Glove, Fasttext, Bert, and Doc2Vec embeddings also...
# This is just a placeholder for a grid search of many of these approaches. 
# It's such an interesting area! 


# Since it's so big, I am going to shrink the wordvec to the relevant words, just for the purposes of this exercise.
# Of course I would keep the full vec really for future test and live cases..

# Note, I am not convinced that 'title' words carry extra category meaning, in fact quite the opposite. I will just concatenate text and title and treat altogether.

### *** Digression 2: Pruning the word Vec



#redo with new set... that tiny translation case..

data['url_words'] = data['url'].apply(joinWordsFromURL)

data['text_title'] = data['title'] + ' ' + data['text']
data['text_title_url'] = data['title'] + ' ' + data['text'] + ' ' + data['url_words']



# I use CountVectorizer here for its internal processing and I can add the counts of the words easily after...

# perhaps this really should happen in a cross-validation but this will do for now. Given the aggregation it probably makes very little difference

# I'll try with/without stops - I want to show you something..

ttNoStopCountVectorizer = CountVectorizer(stop_words='english')
ttNoStopCounts = ttNoStopCountVectorizer.fit_transform(data['text_title'])
ttNoStopWords = ttNoStopCountVectorizer.get_feature_names()

ttStopCountVectorizer = CountVectorizer()
ttStopCounts = ttStopCountVectorizer.fit_transform(data['text_title'])
ttStopWords = ttStopCountVectorizer.get_feature_names()

ttuNoStopCountVectorizer = CountVectorizer(stop_words='english')
ttuNoStopCounts = ttuNoStopCountVectorizer.fit_transform(data['text_title_url'])
ttuNoStopWords = ttuNoStopCountVectorizer.get_feature_names()

ttuStopCountVectorizer = CountVectorizer()
ttuStopCounts = ttuStopCountVectorizer.fit_transform(data['text_title_url'])
ttuStopWords = ttuStopCountVectorizer.get_feature_names()


# countVectorizer.fit_transform(data['text'])
# textWords = countVectorizer.get_feature_names()
# 
# textWords
# 
# titleWords
# ... '你食咗飯沒啊', '唔該', '多謝', 'ꞌgood', 'ﬁre', 'ﬁrm', 'ﬁrst'] ... hmmm I will just 

# textTitleWords = list(set(titleWords).union(textWords))

wordVec = KeyedVectors.load_word2vec_format(fullWordVecFileName, binary=False)
wordVec.vocab

# # https://github.com/RaRe-Technologies/gensim/issues/1882
# ttNoStopSubsetWordVec = subsetW2v(wordVec, ttNoStopWords, dim = 100)
# ttStopSubsetWordVec = subsetW2v(wordVec, ttStopWords, dim = 100)
# ttuNoStopSubsetWordVec = subsetW2v(wordVec, ttuNoStopWords, dim = 100)
# ttuStopSubsetWordVec = subsetW2v(wordVec, ttuNoStopWords, dim = 100)
# 
# ttNoStopSubsetWordVec.save_word2vec_format(ttNoStopSubsetWordVecFileName, binary=False)
# ttStopSubsetWordVec.save_word2vec_format(ttStopSubsetWordVecFileName, binary=False)
# ttuStopSubsetWordVec.save_word2vec_format(ttuStopSubsetWordVecFileName, binary=False)
# ttuNoStopSubsetWordVec.save_word2vec_format(ttuNoStopSubsetWordVecFileName, binary=False)

# ttSubsetWordVec = KeyedVectors.load_word2vec_format(ttSubsetWordVecFileName, binary = False)
# ttuSubsetWordVec = KeyedVectors.load_word2vec_format(ttuSubsetWordVecFileName, binary = False)





## Note, summing and aggregating wordvectors is quite meaningful, regardless of doc2vec etc..
## Indeed it's the very basis of the CBOW model - the centre word is modeled on the aggregation of the context words...

## we could try weighting by TfIDF as per e.g. here https://datascience.stackexchange.com/questions/24855/weighted-sum-of-word-vectors-for-document-similarity
## But I will keep that as a future potential enhancement...




ttStopSentenceVecs = getSentenceMeanWv(ttStopCounts, ttStopWords, wordVec, verbose = True)
"zealand" in ttStopSubsetWordVec.vocab
ttStopSubsetWordVec['zealand']

ttNoStopSentenceVecs = getSentenceMeanWv(ttNoStopCounts, ttNoStopWords, wordVec, verbose = True)

ttuStopSentenceVecs = getSentenceMeanWv(ttuStopCounts, ttuStopWords, wordVec, verbose = True)
ttuNoStopSentenceVecs = getSentenceMeanWv(ttuNoStopCounts, ttuNoStopWords, wordVec, verbose = True)
sentenceVecs.shape
#(8646, 100)



    

ttStopSentenceVecs    

ttNoStopSentenceVecs 

ttuStopSentenceVecs 

ttuNoStopSentenceVecs 


# first of all just try this latent feature set.
# the url I will try to skip, it's either rules or not...


nonZeroCountsLocs[0]

ttCounts[0,0]

ttCounts.sum(axis = 1).min()
 # 29

len(nonZeroCountsLocs[0])
# 2449446

### ****** Can I test it?
### Just make double sure I can match test form to the training
# Yes, CountVectorizer() can take a preexisting vocabulary.
# That's so important for this approach. It always is necessary formally but here we would have completely mismatching vectors otherwise...
testCv = CountVectorizer(vocabulary = ttNames, stop_words='english')
testCounts = testCv.fit_transform(testCases['text_title'])

# I will ue this in the predictor ^^^^



### *** Text
# Include text and title
# I won't look into this much, I'll just decide to build tfidf for text and title, 
# Use this along wordembeddings for title and for text+title
# And then the Url features ... the most empirically interesting I think... 

### *** URL
# I won't use ANY rules here - I think I could easily for some science, but let's try a more general approach..




### ******* Model Training



Metrics



scoreDict = {'accuracy' : 'accuracy', 'kappa' : metrics.make_scorer(metrics.cohen_kappa_score)}


ttStopRf = RandomForestClassifier()
ttNoStopRf = RandomForestClassifier()
ttuStopRf = RandomForestClassifier()
ttuNoStopRf = RandomForestClassifier()

np.isnan(ttSentenceVecs).sum()
# 0
np.isfinite(ttSentenceVecs).all()
# True

np.isnan(ttuSentenceVecs).sum()
# 0
np.isfinite(ttuSentenceVecs).all()
# True

# very important to use a StratifiedKFold in my opinion
ttStopStratKFold = StratifiedKFold(n_splits = 10, shuffle=True, random_state=0)
ttStopStratKFold.get_n_splits(ttStopSentenceVecs, data['category'])
ttStopCvRes =  cross_validate(
        estimator = ttStopRf, 
        X = ttStopSentenceVecs,  
        y = data['category'], 
        cv=ttStopStratKFold, 
        scoring = scoreDict,
        n_jobs=4, 
        verbose = 5)
pd.DataFrame(ttStopCvRes['test_accuracy']).describe()
# count  10.000000
# mean    0.845706
# std     0.014942
# min     0.821759
# 25%     0.838298
# 50%     0.846243
# 75%     0.852141
# max     0.868208

pd.DataFrame(ttStopCvRes['test_kappa']).describe()
# count  10.000000
# mean    0.828568
# std     0.016651
# min     0.801756
# 25%     0.820355
# 50%     0.829199
# 75%     0.835573
# max     0.853625


ttNoStopStratKFold = StratifiedKFold(n_splits = 10, shuffle=True, random_state=0)
ttNoStopStratKFold.get_n_splits(ttNoStopSentenceVecs, data['category'])
ttNoStopCvRes =  cross_validate(
        estimator = ttNoStopRf, 
        X = ttNoStopSentenceVecs,  
        y = data['category'], 
        cv=ttNoStopStratKFold, 
        scoring = scoreDict,
        n_jobs=4, 
        verbose = 5)
pd.DataFrame(ttNoStopCvRes['test_accuracy']).describe()
# count  10.000000
# mean    0.853919
# std     0.010085
# min     0.839120
# 25%     0.848223
# 50%     0.852516
# 75%     0.857184
# max     0.875145
pd.DataFrame(ttNoStopCvRes['test_kappa']).describe()
# count  10.000000
# mean    0.837683
# std     0.011210
# min     0.821194
# 25%     0.831370
# 50%     0.836138
# 75%     0.841249
# max     0.861248



ttuStopStratKFold = StratifiedKFold(n_splits = 10, shuffle=True, random_state=0)
ttuStopStratKFold.get_n_splits(ttuStopSentenceVecs, data['category'])
ttuStopCvRes =  cross_validate(
        estimator = ttuStopRf, 
        X = ttuStopSentenceVecs,  
        y = data['category'], 
        cv=ttuStopStratKFold, 
        scoring = scoreDict,
        n_jobs=4, 
        verbose = 5)
pd.DataFrame(ttuStopCvRes['test_accuracy']).describe()
# count  10.000000
# mean    0.837683
# std     0.011210
# min     0.821194
# 25%     0.831370
# 50%     0.836138
# 75%     0.841249
# max     0.861248
pd.DataFrame(ttuStopCvRes['test_kappa']).describe()
# count  10.000000
# mean    0.834711
# std     0.014010
# min     0.816384
# 25%     0.822545
# 50%     0.834949
# 75%     0.843898
# max     0.860100


ttuNoStopStratKFold = StratifiedKFold(n_splits = 10, shuffle=True, random_state=0)
ttuNoStopStratKFold.get_n_splits(ttuNoStopSentenceVecs, data['category'])
ttuNoStopCvRes =  cross_validate(
        estimator = ttuNoStopRf, 
        X = ttuNoStopSentenceVecs,  
        y = data['category'], 
        cv=ttuNoStopStratKFold, 
        scoring = scoreDict,
        n_jobs=4, 
        verbose = 5)
pd.DataFrame(ttuNoStopCvRes['test_accuracy']).describe()
# count  10.000000
# mean    0.857158
# std     0.009274
# min     0.843750
# 25%     0.853009
# 50%     0.855491
# 75%     0.857803
# max     0.875145
pd.DataFrame(ttuNoStopCvRes['test_kappa']).describe()
# count  10.000000
# mean    0.841288
# std     0.010323
# min     0.826346
# 25%     0.836622
# 50%     0.839538
# 75%     0.842009
# max     0.861309


### For now we'll just go with the last one.
parameters = {
        'bootstrap': [True, False],
        'max_depth': [80, 90, 100, 110],
        'min_samples_leaf': [3, 4, 5],
        'n_estimators' : [50, 100, 150, 200]
        }

rfc = RandomForestClassifier()

randomSearch = RandomizedSearchCV(
    rfc,
    parameters,
    verbose=50, 
    n_iter=10, 
    n_jobs=3, 
    refit = 'accuracy',
    scoring=scoreDict)
randomSearch.fit(ttuNoStopSentenceVecs, data['category'])

print(randomSearch.best_score_)
# 0.8576222682094299

# Not much improvment over the default..
# 
print(randomSearch.cv_results_)
randomSearch.cv_results_.keys()

pd.DataFrame(randomSearch.cv_results_['mean_test_accuracy']).describe()
# count  10.000000
# mean    0.851863
# std     0.004312
# min     0.843397
# 25%     0.849585
# 50%     0.852361
# 75%     0.854066
# max     0.857622


pd.DataFrame(randomSearch.cv_results_['mean_test_kappa']).describe()
# count  10.000000
# mean    0.835348
# std     0.004798
# min     0.825928
# 25%     0.832803
# 50%     0.835923
# 75%     0.837761
# max     0.841757


# It did not beat the default... I could do a bigger default, but default parameters are often robust on certain data sets in my experience...


# Let's see how much more joice xgboost can squeeze out of it...
xgbDefaultModel = xgb.XGBClassifier()

xgbDefaultModelRes =  cross_validate(
        estimator = xgbDefaultModel,
        X = ttuNoStopSentenceVecs,  
        y = data['category'], 
        cv=ttuNoStopStratKFold, 
        scoring = scoreDict,
        n_jobs=4, 
        verbose = 5)
# some deprecation warnings... fine for now...


pd.DataFrame(xgbDefaultModelRes['test_accuracy']).describe()
# count  10.000000
# mean    0.873467
# std     0.013836
# min     0.854167
# 25%     0.863873
# 50%     0.870949
# 75%     0.882659
# max     0.898266

pd.DataFrame(xgbDefaultModelRes['test_kappa']).describe()
# count  10.000000
# mean    0.859560
# std     0.015376
# min     0.838193
# 25%     0.848872
# 50%     0.856643
# 75%     0.869896
# max     0.887061

######################################################################
# Conclusion
######################################################################

### ****** Conclusion
# All of the steps here are really just illustrative of a larger feature search I would perform.
# I focused on word embeddings because it's kind of a pure approach theoretically, and also it transposes the data into a form suitable for more traditional methods like RF and XGB (as opposed to MultinomialNB or SVM we associate with count data.)
# I tried 4 different parameterizations - with/without stop words, with/without URL data.
# Even concatenating the URL data gave a small improvement.
# So no doubt generating rules, or empirically derived domain features, would further improve performance.
# For instance, I would have a feature such as tfidf for certain words, like business etc..
# What I like about my approach is that it lends itself more easily to multiple languages.
# As we saw with the single Hungarian case (and this is why I did it) we can just machine-translate the text and then apply the embeddings approach.
# So it generalizes to more languages, and to more categories, without pernickety feature engineering.
# Now, I actually like, even sometimes love, pernickety feature engineering (it brings me close to the data and to the business problem) but that's not what I wanted to demonstrate today.
# I would consider this model to be a kind of baseline for more empirical approaches.
# 1. It's a pretty good score, probably effective to deploy as MVP if the data matches the domain.
# 2. It's always generalizable to new categories and languages - it's robuust.
# That said, google translation has become a world of pain. I remember the good old days when I was translating several million takeaway menus on ec2 just using a try/catch for when it temporarily blocked. 
# I will now train the model.
# I will just use the default random forest because it's so robust and quick compared to XGB.
# For future steps I would add more rules/empirical features etc. (e.g. searching for certain words in text and URL (separate features for each perhaps). 
# I would also consider other weightings like tfidf.
# Note here I summed all repeat occurrences of words before dividing my total tokens, on an UNNORMALIZED VECTOR. Plenty of things to play about it with here:
# 1. Different word2vec parameterizations, especially cbow - given I do its aggregation basically in the mean. I used the wikipedia2vec embeddings. There are many other ones and different sources (wsj??), I just picked one more or less at random - to demonstrate the robustness of the approach as much as anything.
# 2. Different projection: in particaular I have had good experience with a cholesky decomposition (I think that's what it is or similar - been while since school). I saw a short paper on it back in 2014 for a Netflix challenge algorithm, but cannot find it now, but have used this approach on product data (product vectors). Basically I solve for wordVec * docVec' = wordCounts. 
# ie. W = wordVec, B = wordCounts, D = docVec (or docVec^T) to find, then:
# Assume W.D = B
# W^t.W.D = W^t.B
# (W^t.W)^{-1}.(W^t.W).D = (W^t.W)^{-1}.W^t.B
# D = (W^t.W)^{-1}.W^t.B
# And finally we use E=D^T, i.e. a projection into the space of W, a row per document and the colmun dimension of the word vector.
# i.e. We form a symmetric matrix out of the word vector, and use its inverse to formulate the equation in terms of D, using linear algebra e.g. np.linalg.solve() 
# It's a very elegant approach. I would have liked to demonstrate it but I found an underlying bug in my gensim functions I use to reduce the word vecs per vocabs. I will redo this myself when I get a chance - I'm going to just rip out the vectors and use a named index, dispense with the gensim models entirely, because some of their backend code and model representation has changed over the years, it happens. I'll write something that just works in np over the next days - it's more implementation independent anyway. I made a mistake in my approach there.
# Although, in practice the mean sentence vectors often work just as well (seems data dependent) and they are also theoretically sound - they are the basis of cbow (the middle word is compared to the aggregate of the window words), and something quite similar happens iteratively (over pairwise across the window) in skipgram, I think it's fair to say.
# 2. GlOvE, fasttext, Bert etc... and their parameterizations (e.g. fasttext substrings...)
# 3. Tf-Idf weightings, summation of normalized etc..
# 4. Training specific embeddings when the data is large enough (probably not needed - unless using supervised fasttext where we train targeted at the categories themselves.
# But basically I like the purity of this approach.
#
# I would be optimistic that ifidf might lead to improvement, or anything that accounts for the global or prior probability of a given word. I've seen examples of tdidf embedding weighting around, it's a nice idea for certain tasks. The clue it might work here is that stop word removal helped already, since it kind of denoised the vectors - it's a guess but the improvement was to be expected, and I wanted to demonstrate it, the noise effect of stopwords is quite intuitive when we imagine them being components of a mean. This is probably where I would continue to explore after adding the cholesky part, since I want that code anyway. 
# Note on experience, I tried such weighted product vectors at So1, e.g. downcounting for more common products like bread or beer. But there it was not so useful, because we were very interested in what common products people liked - we offered promotions on them also.
# But for a classification task such as this it's more likely to have a major benefit.
#
# Also I just used CountVectorizer() for implicit tokenization and cleaning.

# More could be done here, but the resultant form makes it suitable for efficient word vector aggregation.
#
# I will build a training model using no stop words with the combined text, title, url, and use the default random forest. I like it's so simple and does quite well.
### ****** Conclusion
# All of the steps here are really just illustrative of a larger feature search I would perform.
# I focused on word embeddings because it's kind of a pure approach theoretically, and also it transposes the data into a form suitable for more traditional methods like RF and XGB (as opposed to MultinomialNB or SVM we associate with count data.)
# I tried 4 different parameterizations - with/without stop words, with/without URL data.
# Even concatenating the URL data gave a small improvement.
# So no doubt generating rules, or empirically derived domain features, would further improve performance.
# For instance, I would have a feature such as tfidf for certain words, like business etc..
# What I like about my approach is that it lends itself more easily to multiple languages.
# As we saw with the single Hungarian case (and this is why I did it) we can just machine-translate the text and then apply the embeddings approach.
# So it generalizes to more languages, and to more categories, without pernickety feature engineering.
# Now, I actually like, even sometimes love, pernickety feature engineering (it brings me close to the data and to the business problem) but that's not what I wanted to demonstrate today.
# I would consider this model to be a kind of baseline for more empirical approaches.
# 1. It's a pretty good score, probably effective to deploy as MVP if the data matches the domain.
# 2. It's always generalizable to new categories and languages - it's robuust.
# That said, google translation has become a world of pain. I remember the good old days when I was translating several million takeaway menus on ec2 just using a try/catch for when it temporarily blocked. 
# I will now train the model.
# I will just use the default random forest because it's so robust and quick compared to XGB.
# For future steps I would add more rules/empirical features etc. (e.g. searching for certain words in text and URL (separate features for each perhaps). 
# I would also consider other weightings like tfidf.
# Note here I summed all repeat occurrences of words before dividing my total tokens, on an UNNORMALIZED VECTOR. Plenty of things to play about it with here:
# 1. Different word2vec parameterizations, especially cbow - given I do its aggregation basically in the mean. I used the wikipedia2vec embeddings. There are many other ones and different sources (wsj??), I just picked one more or less at random - to demonstrate the robustness of the approach as much as anything.
# 2. Different projection: in particaular I have had good experience with a cholesky decomposition (I think that's what it is or similar - been while since school). I saw a short paper on it back in 2014 for a Netflix challenge algorithm, but cannot find it now, but have used this approach on product data (product vectors). Basically I solve for wordVec * docVec' = wordCounts. 
# ie. W = wordVec, B = wordCounts, D = docVec (or docVec^T) to find, then:
# Assume W.D = B
# W^t.W.D = W^t.B
# (W^t.W)^{-1}.(W^t.W).D = (W^t.W)^{-1}.W^t.B
# D = (W^t.W)^{-1}.W^t.B
# And finally we use E=D^T, i.e. a projection into the space of W, a row per document and the colmun dimension of the word vector.
# i.e. We form a symmetric matrix out of the word vector, and use its inverse to formulate the equation in terms of D, using linear algebra e.g. np.linalg.solve() 
# It's a very elegant approach. I would have liked to demonstrate it but I found an underlying bug in my gensim functions I use to reduce the word vecs per vocabs. I will redo this myself when I get a chance - I'm going to just rip out the vectors and use a named index, dispense with the gensim models entirely, because some of their backend code and model representation has changed over the years, it happens. I'll write something that just works in np over the next days - it's more implementation independent anyway. I made a mistake in my approach there.
# Although, in practice the mean sentence vectors often work just as well (seems data dependent) and they are also theoretically sound - they are the basis of cbow (the middle word is compared to the aggregate of the window words), and something quite similar happens iteratively (over pairwise across the window) in skipgram, I think it's fair to say.
# 2. GlOvE, fasttext, Bert etc... and their parameterizations (e.g. fasttext substrings...)
# 3. Tf-Idf weightings, summation of normalized etc..
# 4. Training specific embeddings when the data is large enough (probably not needed - unless using supervised fasttext where we train targeted at the categories themselves.
# But basically I like the purity of this approach.
#
# Other steps like stemming etc might help but I think they might lose information, e.g. I expect more latinate word endings, ization etc, in scientific articles, less in tabloid gossip columns, etc... by aggregating through the word embedding we capture such semantic structure (is I think a well grounded theory. :) )
#
# I would be optimistic that ifidf might lead to improvement, or anything that accounts for the global or prior probability of a given word. I've seen examples of tdidf embedding weighting around, it's a nice idea for certain tasks. The clue it might work here is that stop word removal helped already, since it kind of denoised the vectors - it's a guess but the improvement was to be expected, and I wanted to demonstrate it, the noise effect of stopwords is quite intuitive when we imagine them being components of a mean. This is probably where I would continue to explore after adding the cholesky part, since I want that code anyway. 
# Note on experience, I tried such weighted product vectors at So1, e.g. downcounting for more common products like bread or beer. But there it was not so useful, because we were very interested in what common products people liked - we offered promotions on them also.
# But for a classification task such as this it's more likely to have a major benefit.
#
# Also I just used CountVectorizer() for implicit tokenization and cleaning.

# More could be done here, but the resultant form makes it suitable for efficient word vector aggregation.
#
# I will build a training model using no stop words with the combined text, title, url, and use the default random forest. I like it's so simple and does quite well.
