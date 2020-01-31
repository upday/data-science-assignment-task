Upday Data Science Challenge
==============================

Task: Build a model that can automatically categorise articles

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- Summary of file structure and instructions.
    ├── data
    │   ├── processed      <- The cleaned data set for modeling.
    │   └── raw            <- The original data.
    │
    ├── models             <- Trained Logistic Regression Model.
    │                      <- Tdidf Transformer for feature extraction.
    │
    ├── notebooks          <- Jupyter notebooks: 1.0-sh-ds-challenge.
    │
    ├── requirements.txt   <- The libraries required for reproducing the environment.
    │                         
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │   └── spacy_cleaner.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions.
    │   │   ├── algorith_eval.py 
    │   │   ├── plot_confusion_matrix.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │   
    │   │   
    └──── 
--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.

Instructions for bonus scripts
------------
* cd shaq
* python src/models/train_model to train the data and upload model. This script also contains evaluation metrics for\
 the trained model.
* python src/model/predict_model.py --pass an article to predict category--- to predict the category of a article.\
Insert any length of text, but for now the script can only predict the category for one specific article. 