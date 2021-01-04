import os
import pandas as pd
from time import perf_counter
import spacy
nlp = spacy.load('en_core_web_lg')

def lemmatize_text(text):
    new_text = ''
    for token in nlp(text):
        lemma = token.lemma_
        if token.lemma_ != "-PRON-" and not token.is_punct and not token.is_stop:
            new_text += f" {lemma.lower()}"
    return new_text

def lemmatize_df(df):
    if os.path.isfile('lemmatized_words.csv'):
        df['lemmatized_text'] = pd.read_csv('lemmatized_words.csv')['lemmatized_text']
    else:
        init = perf_counter()
        df['lemmatized_text'] = df.apply(lambda t: lemmatize_text(f'{t.title} {t.text}'), axis=1)
        print(f'Total time: {perf_counter() - init}')
        df.to_csv('lemmatized_words.csv')
    return df

