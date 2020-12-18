from config.config import Config
from model.preprocessing import read_data
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import os

Config.init_config()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to input dataset')
    parser.add_argument('--train', help='path to output train dataset')
    parser.add_argument('--test', help='path to output test dataset')
    parser.add_argument('--size', type=float, help='test dataset size')

    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = os.path.join(Config.get_value("model", "input", "path"), Config.get_value("model", "input", "name"))
    train_output = args.train
    if train_output is None:
        train_output = os.path.join(Config.get_value("data", "split", "train", "path"), Config.get_value("data", "split", "train", "name"))
    test_output = args.test
    if test_output is None:
        test_output = os.path.join(Config.get_value("data", "split", "test", "path"), Config.get_value("data", "split", "test", "name"))
    test_size = args.size
    if test_size is None:
        test_size = Config.get_value("data", "split", "testSize")

    # load file
    df = read_data(input_path)

    # split
    train_df, test_df = train_test_split(df, test_size = test_size, stratify = df.category)

    # save files
    train_df.to_csv(train_output, sep='\t')
    test_df.to_csv(test_output, sep='\t')
