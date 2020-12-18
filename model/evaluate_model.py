import os
import argparse
import fasttext
import pandas as pd
from model.preprocessing import clean_data, read_data, tokenize
from config.config import Config

Config.init_config()

def evaluate_performance(temp_df, threshold):
    data = []
    
    for label in temp_df['category'].drop_duplicates().tolist():
        true_positives = temp_df[(temp_df.predict_label == label) & (temp_df.category == label) & (temp_df.predict_value>=threshold)].shape[0]
        try:
            precision = true_positives/temp_df[(temp_df.predict_label == label) & (temp_df.predict_value>=threshold)].shape[0]
        except:
            precision = 0
        try:
            recall = true_positives/temp_df[temp_df.category == label].shape[0]
        except:
            recall = 0

        try:
            f1 = 2*precision*recall/(precision + recall)
        except:
            f1 = 0
        number = temp_df[temp_df.category == label].shape[0]
        data.append([label, precision, recall, f1, number])
    stat_df = pd.DataFrame(data, columns = ['label', 'precision', 'recall', 'f1', 'count'])
    macrof1 = stat_df['f1'].mean()
    weightedf1 = stat_df['f1'].values.dot(stat_df['count'].values)/stat_df['count'].sum()
    macroprecision = stat_df['precision'].mean()
    weightedprecision = stat_df['precision'].values.dot(stat_df['count'].values)/stat_df['count'].sum()
    macrorecall = stat_df['recall'].mean()
    weightedrecall = stat_df['recall'].values.dot(stat_df['count'].values)/stat_df['count'].sum()
    
    print('Macro F1: {}'.format(macrof1))
    print('Weighted F1: {}'.format(weightedf1))
    print('Macro Precision: {}'.format(macroprecision))
    print('Weighted Precision: {}'.format(weightedprecision))
    print('Macro Recall: {}'.format(macrorecall))
    print('Weighted Recall: {}'.format(weightedrecall))
    
    return stat_df

def predict(model, row, feature_fields):
    clean_description = ''
    for field in feature_fields:
        clean_description += row[field] + ' '
    clean_description = clean_description.replace('\n', ' ')
    clean_description = tokenize(clean_description)
    clean_description = (clean_description + ' ') * 1
    clean_description = clean_description.strip()
    result = model.predict(clean_description,k=3)
    
    return result[0][0][9:], result[1][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='path to test dataset')

    args = parser.parse_args()

    test_input = args.test
    if test_input is None:
        test_input = os.path.join(Config.get_value("data", "split", "test", "path"), Config.get_value("data", "split", "test", "name"))

    # load_model
    model_path = os.path.join(Config.get_value("model", "output", "path"), Config.get_value("model", "output", "name"))
    model = fasttext.load_model(model_path)

    # load and clean the test dataset
    data_df = read_data(test_input)
    # clean data
    data_df = clean_data(data_df)
    # predict using the model
    data_df['predict_label'], data_df['predict_value'] = zip(*data_df.apply(lambda x: predict(model, x, ["website_url", "article_cats", "cleaned_title", "cleaned_text"]), axis=1))

    # evaluate the performance
    threshold = Config.get_value("model", "threshold")
    evaluate_performance(data_df, threshold)