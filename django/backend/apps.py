from django.apps import AppConfig
import joblib
import spacy


class BackendConfig(AppConfig):
    name = 'backend'
    nlp = spacy.load('en_core_web_lg')
    predictor = joblib.load("pipe.joblib")
