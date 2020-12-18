import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def read_data(path):
    df = pd.read_csv(path, sep="\t")
    return df

def combine_text(data, feature_fields):
    """
    Combine text fields in a column, add __label__ to label and concat them to prepare for fasttext training
    """
    data["Text"] = ''
    for field in feature_fields:
        data["Text"] += data[field] + ' '
    data["Label"] = '__label__' + data["category"]
    data["TextWLabels"] = data["Text"] + ' ' + data["Label"]
    return data

# download stopwords if it's not downloaded
nltk.download('stopwords')
ENGLISH_STOPWORDS = set(stopwords.words('english'))
def tokenize(text):
    """
    Clean the text by remove numbers, punctation, stopwords, change to lowercase
    """
    tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')
    token_list = tokenizer.tokenize(text)
    
    return ' '.join([w.lower() for w in token_list if len(w) > 1 and w not in ENGLISH_STOPWORDS])

def tokenize_url(url):
    """
    Clean the url and separate them to website url, article tags and article title
    """
    # for url, fasttest way is to split by /
    tokens = url.split('/')
    if len(tokens) > 3:
        # ignore the http and double slash, the website address is the 3rd
        website_url = tokens[2]
        # the final should contain article title
        article_title = tokens[-1]
        # we should clean the extensions to get the title
        article_title = re.sub('\.[\w]{3,4}$', '', article_title)
        article_title = tokenize(article_title)

        # now the important part, the parts in between. Usually they contain the article categories or date, let's try to ignore numbers
        article_cats = tokenize(' '.join([c for c in tokens[3:-1] if not re.search('\d', c)]))
        return website_url, article_cats, article_title
    else:
        return '', '', tokenize(url)

def clean_data(data_df):
    """
    Clean the text from dataset
    """
    data_df['cleaned_title'] = data_df['title'].apply(tokenize)
    data_df['cleaned_text'] = data_df['text'].apply(tokenize)

    data_df['website_url'], \
    data_df['article_cats'], \
    data_df['article_title'] = zip(*data_df['url'].apply(tokenize_url))

    return data_df