"""
This script evaluates the training and holdout test data based on selected metrics.
Note that the `train_data.py` file NEEDS to be ran beforehand as this file uses outputs generated from the training file.

"""
import pickle
import pandas as pd
import numpy as np
import os

from sklearn.metrics import precision_recall_fscore_support


def get_prediction(clf,X,y,encoder):
	# prediction
	y_pred = clf.predict(X)
	pred_class = encoder.inverse_transform(y_pred)

	# results
	pred_result = pd.DataFrame({'y_pred':pred_class, 'y_true':encoder.inverse_transform(y)})
	return pred_result

def evaluate_results(pred_result,encoder):
	precision, recall, fscore, support = precision_recall_fscore_support(pred_result.y_true,pred_result.y_pred)
	metrics_df = pd.DataFrame({'class':encoder.classes_,
                        'precision':precision,
                        'recall':recall,
                        'fscore':fscore,
                        'support':support})
	return metrics_df

def main():
	# load model elements
	current_dir = os.path.dirname(__file__)
	mnb = pickle.load(
		open(os.path.join(current_dir, 
			"model/mnb_model.pkl"),'rb'))
	y_encoder = pickle.load(
		open(os.path.join(current_dir, 
			"model/y_encoder.pkl"),'rb'))

	# load preprocessed train and test data
	X_train = pd.read_csv(os.path.join(current_dir, "data/processed/eeX_train_processed.csv"))
	X_test= pd.read_csv(os.path.join(current_dir, "data/processed/X_test_processed.csv"))
	y_train = np.load(os.path.join(current_dir,"data/processed/y_train_enc.npy"))
	y_test = np.load(os.path.join(current_dir,"data/processed/y_test_enc.npy"))

	# get predictions for train and test
	train_pred = get_prediction(mnb,X_train,y_train,y_encoder)
	test_pred = get_prediction(mnb,X_test,y_test,y_encoder)
	print(test_pred)

	# get evaluation metrics for train and test
	train_metrics = evaluate_results(train_pred,y_encoder)
	test_metrics = evaluate_results(test_pred,y_encoder)

	print(f"Training data evaluation metrics:")
	print(train_metrics)

	print(f"\n\nTest data (Holdout) evaluation metrics:")
	print(test_metrics)


if __name__ == "__main__":
	main()