import argparse
from model.preprocessing import read_data, combine_text, clean_data
from config.config import Config
import os
import csv
from pathlib import Path
import fasttext

Config.init_config()

def train_model(train_data , wordNgrams, lr, epoch, feature_fields, dim=100, pretrained_vectors = None):
    """
    Train a model by fasttext based on processed train data
    """
    train_data = combine_text(train_data, feature_fields)
    train_data = train_data[train_data['TextWLabels'].notnull()]
    train_dataframe = train_data[['TextWLabels']]
    
   
    train_dataframe.to_csv('training_file.csv', sep='\t', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE, escapechar='"', header=False)
     
    if pretrained_vectors is None:
        model = fasttext.train_supervised('training_file.csv', wordNgrams=wordNgrams, lr=lr, epoch=epoch, dim=dim)
    else:
        model = fasttext.train_supervised('training_file.csv', wordNgrams=wordNgrams, lr=lr, epoch=epoch, dim=dim, pretrainedVectors=pretrained_vectors)
    
    os.remove('training_file.csv') 

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to input train data')
    parser.add_argument('--output', help='path to output model')

    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = os.path.join(Config.get_value("model", "input", "path"), Config.get_value("model", "input", "name"))
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(Config.get_value("model", "output", "path"), Config.get_value("model", "output", "name"))

    # read data
    data_df = read_data(input_path)
    # clean data
    data_df = clean_data(data_df)
    # train model
    wordngrams = Config.get_value("model", "trainParams", "wordngrams")
    lr = Config.get_value("model", "trainParams", "lr")
    epoch = Config.get_value("model", "trainParams", "epoch")
    model = train_model(data_df, wordngrams, lr, epoch, ["website_url", "article_cats", "cleaned_title", "cleaned_text"])
    # save to output folder
    # create folder if not exist
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    # save 
    model.save_model(output_path)
