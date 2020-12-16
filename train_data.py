"""
Script to preprocess data and train model
"""
import os
import numpy as np
import pandas as pd
import pickle

from nltk.corpus import stopwords
from string import punctuation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate



def clean_text(text,all_puncs,incl_puncs,stopwords):
	# remove pun
    for pun in list(set(all_puncs) - set(incl_puncs)):
        if pun in text:
            text = text.replace(pun,"")
    # add white space to allowed punctuations for tokenization
    for pun in incl_puncs:
        if pun == '$':
            text = text.replace(pun, "$ ")
        else:
            text = text.replace(pun, f" {pun}")
    
    # lower case & tokenize & stopwords
    return [t for t in text.lower().split() if t not in stopwords]

def clean_data(data):
	all_puncs = list(punctuation) + ['’','‘']
	incl_puncs = ['$','%','!','?']

	sw = list(stopwords.words('english'))
	for pun in list(set(all_puncs) - set(incl_puncs)):
	    sw = [i.replace(pun,'') for i in sw]

	for t in ['title','text']:
	    data[f'{t}_cleaned'] = [clean_text(i,all_puncs,incl_puncs,sw) for i in data[t]]
	    data[f'{t}_cleaned_len'] = [np.log(len(i)+1) for i in data[f'{t}_cleaned']]
	    data[f'{t}_cleaned_unique_len'] = [np.log(len(np.unique(i))+1) for i in data[f'{t}_cleaned']]
	    data.drop(t,axis=1,inplace=True)

	# joining back text data as single string
	for t in ['title_cleaned','text_cleaned']:
	    data[t] = [' '.join(i) for i in data[t]]

	return data


def main():
	current_dir = os.path.dirname(__file__)
	# read data
	data = pd.read_csv(os.path.join(current_dir, "data/raw/data_redacted.tsv"),
		delimiter='\t')
	print(f"Dataframe shape: {data.shape}")

	# split data to train and test for final evaluation
	X_train, X_test, y_train, y_test = train_test_split(data.drop(['category','url'], axis=1)
			,data.category,train_size=0.9
			,random_state = 2020
			,stratify=data.category)
	print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

	# clean data
	X_train_cleaned = clean_data(X_train)
	X_test_cleaned = clean_data(X_test)

	# vectorize text data
	vectorizer = TfidfVectorizer(analyzer='word',max_features=7000, min_df=0.01)
	X_train_vecs = pd.DataFrame(vectorizer.fit_transform(X_train_cleaned.title_cleaned + " " + X_train_cleaned.text_cleaned).toarray())
	X_test_vecs = pd.DataFrame(vectorizer.transform(X_test_cleaned.title_cleaned + " " + X_test_cleaned.text_cleaned).toarray())
	# assign original index to concat with other data
	X_train_vecs.index = X_train_cleaned.index
	X_test_vecs.index = X_test_cleaned.index

	# concat vectorized text data with other features
	X_train_full = pd.concat([X_train_vecs,X_train_cleaned[['title_cleaned_len','title_cleaned_unique_len','text_cleaned_len','text_cleaned_unique_len']]],axis=1)
	X_test_full = pd.concat([X_test_vecs,X_test_cleaned[['title_cleaned_len','title_cleaned_unique_len','text_cleaned_len','text_cleaned_unique_len']]],axis=1)
	
	# encode target
	y_encoder = LabelEncoder()
	y_train_enc = y_encoder.fit_transform(y_train)
	y_test_enc = y_encoder.transform(y_test)

	print(f"Dataframe shape after preprocessing: Train: {X_train_full.shape}, Test: {X_test_full.shape}")

	# cross validate and train
	print(f"Training classification model...")
	mnb = MultinomialNB(alpha =0.1)
	cv_cnt = cross_validate(mnb, X_train_full, y_train_enc, cv=20,
	                    scoring=('accuracy','precision_weighted','recall_weighted','f1_weighted'),
	                    return_train_score=True)
	print("Training and validation results:")
	print(pd.DataFrame(cv_cnt).describe())
	mnb.fit(X_train_full,y_train_enc)

	# save model, vectorizer, encoder
	pickle.dump(vectorizer, 
		open(os.path.join(current_dir, 
			"model/vectorizer.pkl"),'wb'))
	pickle.dump(y_encoder,
		open(os.path.join(current_dir, 
			"model/y_encoder.pkl"),'wb'))
	pickle.dump(mnb,
		open(os.path.join(current_dir, 
			"model/mnb_model.pkl"),'wb'))

	# save data train and test data
	X_train_full.to_csv(os.path.join(current_dir, 
		"data/processed/X_train_processed.csv"),index=False)
	X_test_full.to_csv(os.path.join(current_dir,
		"data/processed/X_test_processed.csv"),index=False)
	np.save(open(os.path.join(current_dir,
		"data/processed/y_train_enc.npy"),'wb'),y_train_enc)	
	np.save(open(os.path.join(current_dir,
		"data/processed/y_test_enc.npy"),'wb'),y_test_enc)	

if __name__ == "__main__":
	main()

