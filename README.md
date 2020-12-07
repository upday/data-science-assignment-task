# Solution

The notebook detailing the solution: classification.ipynb
~A report about the data can be found at ./data/basic_analysis.html~ On second thoughts, the report can be generated through the script in the notebook and is not committed to github.

## Training the model:
Assuming a TSV in ./data/ named df_train (can also be generated using the nb), or any other data file in the right format. <br>
```python3 text_classification.py train_model ./data/df_train.tsv```   

## Evaluating the model:  
Aforementioned assumptions.<br> 
```python3 text_classification.py test_predictions ./data/df_test.tsv```

## Predicting categories for unseen data:  
(Replace df_test with target file w/o categories). <br>
```python3 text_classification.py test_predictions ./data/df_test.tsv```
  
Pickled models could not be commited to github, you will have to train the model again(shouldn't take long)

---

# Assignment Task for Data Scientists

## Introduction
You are a Data Scientist working at upday, a news aggregator and recommender.

The engineering team at upday is gathering on a regular basis articles from all the Web. In order to provide a proper filtering functionality in the app, the articles need to be categorized.

You have at your disposal a pre-labelled dataset that maps different articles and their metadata to a specific category.

It's up to you now to help the company providing a solution for automatically categorizing articles.

## Assignment
The repository contains a dataset with some english articles and some information about them:

* category
* title
* text
* url

The purpose of the task is to provide a classification model for the articles.

## Instructions

You should make a pull request to this repository containing the solution. If you need to clarify any point of your solution, please include an explanation in the PR description.

What we expect:

* Explanation about the solution you adopted and the results from your data exploration
* Documentation of the results of your model, including the metrics adopted and the final evaluation
* The training and evaluation code

The solution should just perform better than random, also we expect you to use model that is not just rules-based.

How to present the documentation and the code is up to you, whether to provide one or more jupyter notebooks or via a different mean.

## Bonus
Scripts to be run from the command line:

* A script for training the dataset
* A script for evaluating the dataset
* A script to infer the category given an article

