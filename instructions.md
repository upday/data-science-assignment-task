# Readme
This repo contains py scripts and tools to perform a "News Classification".

Entry point:
- **textclf/pipeline_complete.py** contains all the main steps of the pipeline
    - dataset creation
    - training
    - evaluation

Moreover there are other secondary entry points required by yor **Readme.md**. For more details go to section **Usage of the Repo**.

## Requirements
- linux (tested on Ubuntu 20.04)
- conda

## Install
- Setup environment
    - launch the following commands:
        - `conda env create -f environment.yaml --force`
        - `conda activate textclf`
        - `pip install -e .`
    - extract zip data in folder **data**
        - **data_redacted.tsv** must be in folder **data**

## Usage of the Repo
There are 1+3 main scripts:
- pipeline_datasets.py
    - A script for creating train and test datasets (split 80%-20% stratifying on "category")
    - It will take a .tsv file as well as the destination paths of train and test datasets I will create after the split 
    - Example: `python textclf/pipeline_datasets.py --path_data "data/data_redacted.tsv" --path_train "data/dataset_train_raw.p" --path_test "data/dataset_test_raw.tsv"`
- pipeline_training.py
    - A script for training the dataset
    - Example: `python textclf/pipeline_training.py --path_train "data/data_train_raw.tsv"`
- pipeline_evaluation.py
    - A script for evaluating the dataset
    - Example: `python textclf/pipeline_evaluation.py --path_data "data/data_test_raw.tsv"`
- pipeline_inference.py
    - A script to infer the category given an article
    - Example: `python textclf/pipeline_inference.py --path_data "data/dataset_test_raw.tsv" --path_result "data/output.tsv"`

## Description of the Workflow

To approach the problem, an exploratory analysis is mandatory: I must see what is the data I have to handle and predict in order to find the best (hopefully) strategy. 
### Recap Exploratory
A Notebook of exploratory analysis can be inspected 
- open jupter environment 
    - launch in terminal `jupyter notebook`
    - then open **notebooks/exploratory_analysis.ipynb**

Exploratory analysis gave some insights I need to consider for the algorithm I want to use.
- ~ 9'000 records
- no null values or duplicates
- "category" quite umbalanced (but not too drammatically)
- "text" and "title" can be very long
- num tokens in "text" and "title" can be useful for "category" prediction  
- url can potentially contain useful info
- netloc (domain of url) can be considered as a categorical feature
- a few netlocs appear many times, the other are not so frequent 

### Overall Algorithm

First of all, some notes about the **y** column. 
Since it is categorical and quite umbalanced, I think that the metric I wanto to use are
- f1_score
- accuracy

Since the number of records is not huge (and I don't have lots of computing power + time), I decided not to use Deep Learning Techniques in general. Even if, I think that it would be very interesting to see how a Fine-Tuning using Bert (or similar) can perform, see also **Improvements** below.

Therefore, I decided to go with a more traditional approach:
    - extract some features
        - I extract domains from **url** columns that I call **netloc**. Netlocs count is distributed almost like a power-law: so i keep the most frequent names, the rest is mapped into *'Other'*
        - **path** from **url** that will be a new text column
    - cleaning text columns
        - remove stopwords
        - remove punct
        - truncate long text (i keep the first N=512 tokens of "text" and "title") since I assume that the news category can be inferred just with the first tokens...
        - ...
        - see **textclf/preprocessing/traditional_text_preprocessing.py** for more details
    - apply **tf-idf** and **tf**
    - [OPTIONAL] find spacy embeddings with the text column cleaned at the previous step (see flags *use_spacy* in **textclf/pipeline_training.py**)
        - for each text value of a column (that is a sentence), i take the avg of the embeddings of all the tokens of that sentence
    - [OPTIONAL] find bert embeddings with the original text columns (see flags *use_bert* in **textclf/pipeline_training.py**)

Finally there is a hyper-tuning part (exploiting the RandomizedSearchCV by sklearn) over a LightGBoost classifier.

SPOILER metrics:
- TRAIN
    - acc ~ 0.85
    - f1 ~ 0.85
- TEST
    - acc ~ 0.85
    - f1 ~ 0.85  

When launching evaluating script there is also the confusion matrix.

I guess that it would be pretty easy to reach 0.9 accuracy playing more with 
- the spacy+bert encodings 
- hypertuning
- stacking classifier

Just a final note: I didn't consider at all constraints on execution time, resources, ... Those would probably change the chosen strategy.

## Notes
- black formatting (https://github.com/psf/black)
- logs are written in folder `/logs/`
- datasets are saved and loaded as deafult from `/data/`
- models are saved and loaded as deafult from `/models/`
- **requirements.txt** is empty since it is needed by **setup.py** but I have a conda env
- time for scripts in my experimets (16GB ram + 8 cpus) with default config
    - dataset creation: 0.1 min
    - training: 10 min (use_spacy=False + use_bert=False + no hypertuning (1 iteration) over ~6'500 records) 
    - evaluation: 0.15 min (~1'700 records)
    - inference: 0.15 min (~1'700 records)

## Main Problems
- languages
    - the current pipeline can only handle english sentences
- length text
    - some records have a very long "text": the number of words can be greater than 35'000.
    I chose to truncate the text and take the first 512 words. However, there are other possible strategies, like
        - split each text in several sentences
        - classify each sentence
        - aggregate result
        - this could make the system more robust + give a sort of additional confidence

## Improvements
Due to the lack of time, lots of things can be improved.
Here a small list:

- "Model" improvements
    - better fine-tuning
        - automated feauture selection. There are multiple way to create embeddings for words
            - tf-idf + bow
            - word embedding with traditional methods (glove, w2v, ...) 
            - "sentence" embedding using transformers
        - find better parameters tf-idf
        - fine tuning bert on another "news classification" dataset (e.g. https://www.kaggle.com/rmisra/news-category-dataset). Then fine tuning again over this dataset.
        - stack different classifiers
    - NER: knowing that in a news there are some dates, names, or similar things is very informative  
    - better text cleaning
        - expand stopwords
- "devops" improvements
    - make a docker image
    - load data in batches thanks to data generator. Not so priority since the dataset is not so big (for now)
    - optimize for GPU if bert component is kept
    - serve the model in a WS (e.g. flask+gunicorn)
    - remove annoying warning from huggingface (pytorch)
        - **UserWarning: CUDA initialization: Found no NVIDIA driver on your system**
