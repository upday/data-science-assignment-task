import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')


# train model with various classifiers and cross-validate
def cross_validate(X_train, y_train):

	cv = 5
	clf_MultinomialNB = MultinomialNB()
	clf_LinearSVC = LinearSVC()
	clf_RandomForestClassifier = RandomForestClassifier()

	clfs = [clf_MultinomialNB, clf_LinearSVC, clf_RandomForestClassifier]

	clfs_eval = pd.DataFrame({'clf_name' : [], 'clf': [], 'score': []})

	for clf in clfs:
		name = str(clf)
		scores = cross_val_score(clf, X_train, y_train, cv=5)
		print("\t- "+name.replace("()","")+" ("+str(round(scores.mean(),2))+")")
		clfs_eval=clfs_eval.append(pd.DataFrame({'clf_name' : [name], 'clf': [clf], 'score': [scores.mean()]}))
	
	best_clf = clfs_eval.sort_values('score', ascending=False).reset_index().clf[0]
	return(best_clf)


# undersample training data
def undersample_random(X_train, y_train):

	y_train = y_train.reset_index()
	sample_size = y_train.category.value_counts().min()

	undersampled_indices = []
	for c in list(y_train.category.unique()):
	    c_index = list(y_train[y_train.category == c].index)
	    c_index_sampled = list(np.random.choice(c_index, size=sample_size, replace=False))
	    undersampled_indices += c_index_sampled
	    
	X_train_undersampled = X_train[undersampled_indices]
	y_train_undersampled = y_train.loc[undersampled_indices]

	return(X_train_undersampled, y_train_undersampled)


## Tokenisation, Stemming, Vectorisation

def stemText(text):
	stemmer = PorterStemmer()
	token_words = word_tokenize(text)
	stem_sentence=[]
	for word in token_words:
		stem_sentence.append(stemmer.stem(word))
		stem_sentence.append(" ")
	return "".join(stem_sentence)


def vectoriseText(corpus):
	vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)[a-zA-Z]+', min_df=0.01, max_df=0.9)
	corpus_stemmed = [stemText(t) for t in corpus]
	corpus_vec = vectorizer.fit_transform(corpus_stemmed)
	vocab = vectorizer.vocabulary_    
	print(corpus_vec.shape)
	return(corpus_vec, vocab)


def main(X_train, y_train, datapath):

	# if vectorised data is available, load it
	if os.path.isfile(datapath+"corpus_vectorised_train.pickle"):
		print("- Loading vectorised corpus (train data) ...")
		f = open(datapath+"corpus_vectorised_train.pickle", "rb")
		corpus_vectorised = pickle.load(f)

	# else, tokenise, stem, vecorise, safe as vector for next time
	else:
		print("- Vectorising corpus (train data) ...")
		corpus_vectorised, vocab = vectoriseText(X_train)
		f1 = open(datapath+"corpus_vectorised_train.pickle", "wb") 
		pickle.dump(corpus_vectorised, f1)
		f1.close()

		f2 = open(datapath+"vocabulary.pickle", "wb") 
		pickle.dump(vocab, f2)
		f2.close()




	# account for unbalanced data, e.g. random undersampling
	print("- Undersampling training data ...")
	X_train_undersampled, y_train_undersampled = undersample_random(corpus_vectorised, y_train)

	# train and cross-valdiate a couple of models, return 'best' model 
	print("- Train and cross-validate models ...")
	clf = cross_validate(X_train_undersampled, y_train_undersampled.category)
	best_classifier = clf.fit(X_train_undersampled, y_train_undersampled.category)
	return(best_classifier)



if __name__ == "__main__":
    main(X_train, y_train, datapath)






