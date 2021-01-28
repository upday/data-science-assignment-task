import os
import pickle
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk


## Tokenisation, Stemming, Vectorisation
def stemText(text):
	stemmer = PorterStemmer()
	token_words = word_tokenize(text)
	stem_sentence=[]
	for word in token_words:
		stem_sentence.append(stemmer.stem(word))
		stem_sentence.append(" ")
	return "".join(stem_sentence)


def vectoriseText(corpus, vocab):
	vectorizer = TfidfVectorizer(vocabulary=vocab)
	corpus_stemmed = [stemText(t) for t in corpus]
	corpus_vec = vectorizer.fit_transform(corpus_stemmed)
	vocab = vectorizer.vocabulary_    
	return(corpus_vec)



def main(X_test, y_test, classifier, datapath):
	
	# if vectorised data is available, load it
	if os.path.isfile(datapath+"corpus_vectorised_test.pickle"):
		print("- Loading vectorised corpus (test data)...")
		f = open(datapath+"corpus_vectorised_test.pickle", "rb")
		corpus_vectorised = pickle.load(f)

	# else, tokenise, stem, vecorise, safe as vector for next time
	else:
		print("- Vectorising corpus (test data) ...")
		f1 = open(datapath+"vocabulary.pickle", "rb")
		vocab = pickle.load(f1)

		corpus_vectorised = vectoriseText(X_test, vocab)
		f = open(datapath+"corpus_vectorised_test.pickle", "wb") 
		pickle.dump(corpus_vectorised, f)
		f.close()


	print("- Predicting class labels with classifier "+str(classifier)+" ...")
	predicted = classifier.predict(corpus_vectorised)
	print("- Classification Report:")
	clf_report = classification_report(y_test, predicted)
	print(clf_report)
	# clf_report_df = pd.DataFrame.from_dict(clf_report, orient='columns').transpose()[:12].sort_values(by='support')[:3]
	# clf_report_df