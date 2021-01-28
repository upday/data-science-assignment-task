import os
import pandas as pd
import pickle
import train_model
import test_model
from sklearn.model_selection import train_test_split



def main(data_path, input_data):
	data = pd.read_csv(data_path+input_data, sep="\t")

	# split dataset into training and test data
	X_train, X_test, y_train, y_test = train_test_split(data.text, data.category, 
	                                                    test_size = 0.25, 
	                                                    stratify = data.category,
	                                                    random_state = 0)

	print("\nTRAINING MODEL\n")
	classifier = train_model.main(X_train, y_train, datapath)
	print("\nTESTING MODEL (or predict categories on new data)\n")
	test_model.main(X_test, y_test, classifier, datapath)


if __name__ == "__main__":
	datapath = "../data/"
	main(datapath, "data_redacted.tsv")




