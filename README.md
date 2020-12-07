# Solution

The notebook detailing the solution can be found here: classification.ipynb<br>
The task is a simple classification task with small tabular data, hence most of the sophisticated neural net approaches(BERT, W2V) are an overkill. Logistic Regression worked best for this task. It was evaluated against some other approaches like XGBoost(,and decision trees, Random Forests), MLP. 
<br>

Evaluation metrics were of the order:
```
LogReg(text+title+url):
Report:
                           precision    recall  f1-score   support

             cars_motors       0.92      0.89      0.91        38
                 culture       0.92      0.92      0.92        50
            digital_life       0.90      0.90      0.90        78
fashion_beauty_lifestyle       0.94      0.94      0.94       125
          money_business       0.79      0.86      0.82        43
                   music       0.97      0.97      0.97        40
                    news       0.91      0.83      0.87        52
            people_shows       0.83      0.75      0.79        32
                politics       0.90      1.00      0.95        26
                  sports       0.98      0.96      0.97        91
      technology_science       0.92      0.93      0.93        74
                  travel       0.92      1.00      0.96        44

                accuracy                           0.92       693
               macro avg       0.91      0.91      0.91       693
            weighted avg       0.92      0.92      0.92       693
```
~A report about the data can be found at ./data/basic_analysis.html~ On second thoughts, the report can be generated through the script in the notebook and is not committed to github.


## Training the model:
Assuming a TSV in ./data/ named df_train.tsv (Note: df_train is a stratified subset of the entire available data, the notebook has code to do so). <br>
```python3 text_classification.py train_model ./data/df_train.tsv```   

Pickled models could not be commited to github, you will have to train the model again(shouldn't take long)<br>
Training one of the available models: <br>
1. [Here](https://github.com/varunchitale/data-science-assignment-task/blob/f4e4c8b3619c1f5d8d718cc594184f33ff3e8415/text_classification.py#L51) are 5 potential models used for classification. In case of retraining, comment out all except one.
2. Run the command for training the model as described above.


## Evaluating the model:  
Aforementioned assumptions for a test dataset, let's call it df_test.<br> 
```python3 text_classification.py test_predictions ./data/df_test.tsv```

## Predicting categories for unseen data:  
df_unseen is w/o categories. <br>
```python3 text_classification.py test_predictions ./data/df_unseen.tsv```




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

