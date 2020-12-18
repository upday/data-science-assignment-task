from model.preprocessing import tokenize, tokenize_url
import fasttext
import os
import argparse
from config.config import Config

Config.init_config()

def predict(model, url, title, text):
    tokenized_url = tokenize_url(url)
    clean_description = ' '.join(tokenized_url) + ' ' + tokenize(title) + tokenize(text)
    clean_description = clean_description.replace('\n', ' ')
    clean_description = tokenize(clean_description)
    clean_description = (clean_description + ' ') * 1
    clean_description = clean_description.strip()
    result = model.predict(clean_description,k=3)
    
    return result[0][0][9:], result[1][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='url of article', default='')
    parser.add_argument('--title', help='title of article', default='')
    parser.add_argument('--text', help='text of article', default='')

    args = parser.parse_args()

    # load_model
    model_path = os.path.join(Config.get_value("model", "output", "path"), Config.get_value("model", "output", "name"))
    model = fasttext.load_model(model_path)

    # predict model
    label, probability = predict(model, args.url, args.title, args.text)

    print(f"Predicted label: {label}")
    print(f"Probability: {probability}")