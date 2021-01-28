# README to my solution to the upday data science assignment

## Intro

The solution I am providing here is not what I would put in production should I be given this task on the job. In practice I would have more product context, especially more insight into what success looks like (i.e. what makes this or that topic/category classification 'best' for our client). IN that context I would optimise preprocessing, the choice of classifier, it's hyperparameters to maximise that metric.In the context of this data science assignment, I am providing one possible solution, which aims to show my thought process and technical ability to execute the approach.

## Containerisation

To run the solution in a docker environment `cd` into `data-science-assignment-task` and run:
- `docker build . -t coding-task:latest` to build
- `docker run -it -v $(pwd)/data:/data/ coding-task` to run the docker container

The folder `/data` in the container syncs with `/data` on local. The input data `data_redacted.tsv` should be placed in the local `/data` for the contained to read the input. All output files (vectorised text data and vocabulary) will be saved to `/data` and will remain on local even when the container shuts down. This will save time for vectorisation (which is an expensive part) when running model re-training (and testing).

To run training and testing consecutively run `python upday_task.py`.

The image contains:
- a `data` folder which syncs with the local 'data' folder containing the unzipped original data as well as vectorised versions of the text, both training corpus and test corpus
- a `source` folder with...
	- a `train_model.py` to train a few classifiers and return the one with the highest score
	- a `test_model.py` to use this classifier for predicting class labels for unseen data (here the test set) and evaluate
	- a `upday_task.py` to run `train_model.py` and `test_model.py` with the given data


## Train-Test Split
The split into training set and test set is done on the raw data (retaining the relative distribution of the class labels, stratification), keeping 25% as a hold out test set. The remaining 75% of the data are used for training and cross-validation.


## Training

### Feature Selection/Text Representation
As a preprocessing step I am stemming the words in the text. Lemmatisation with POS information would be the better choice (I didn't leave time to improve on this).

To represent the text data, I am choosing TF-IDF with unigrams as a BOW approach. Tokenisation and vectorisation is done with scikit-learn and the resulting vocabulary saved for vectorisation of the test data. 

To represent contextual relationships between words, I would in practice use 2-grams or 3-grams and and tune the cut-offs for frequent and rare tokens to strike a balance between performance and classification metric.
On the job I would compare the classification metrics of this BOW representation against word-embeddings (word2vec) and possibly also language models (transformer models). However, the choice of text representation is not only about classification performance but depening on the product needs / stakeholders involved about interpretability of the model. A TF-IDF representation does not provide much contextual relation but is easier to interpret than more complex alternatives.

### Training Data/Class Distribution
Initial exploration of the dataset shows that the classes are not balanced (see Fig1). While any model should be evaluated on the real distribution of classes, there are scanarios where balancing the training data makes sense. 
For this coding test I choose to randmly undersample the larger classes, as the most simple approach. Other options would be the cluster method (for undersampling) or SMOTE, ADASYN (for oversampling). (also implemented in the library imbalanced-learn)
Whether to choose under/oversampling should also be informed by the product requirements. On the job I would try to understand whether it is important to get all the categories right equally or whether majority (or minority) categories have a higher importance for the product. That would inform the sampling technique to use and whether to push for precsion or recall (or both equally).


### Classifiers
I simply choose a handful for classifiers which are commonly used for multiclass text classification. For this execise, I didn't spend a great deal of time choosing the 'right' algorithms or tuning their hyperparameters. Any number of additional classifieres could be added here to the comparison. 
On the job, I would of course perform hyperparameter optimisation, e.g. with methods provided by scikit-learn such as GridSearchCV.

If interpretability is not of importance, I would also try transfer learning, applying an existing pre-trained model to the task. Pre-trained models such as BERT or ELMO have been applied successfully on NLP tasks and may well outperform all 'traditional classifiers'.

### Evaluation/Cross-validation
A number of evaluation metrics are available for multiclass classification. The default one is accuracy but which one to optimse for depends on the product needs. On the job, I would first explore with product owners what we are trying to achieve when a user searches or filters for a certain category. In the context of a news aggregator, the user will likely not notice if the list of articles is 'incomplete' but rather be bothered to see an article that obviously does not belong in the category they are looking for (in this case we would want to maximise precision). For this exercise I simply went with the default 'score' of the model and passed on the 'winning' classifier to `test-model.py`.

## Testing/Calssification of unseen data
After the classifier with the best performace in cross-validation was fitted to the entire (undersampled) training data, it is used to predict class labels on our hold-out test data. The classification report details performance on each individual class. 

In a live scenario, the vectorisation and category prediction of new articles coming in could be ran in parallel, for X batches of articles at a time running a dockerised version of the `test-model.py` alone on AWS Batch for example. 


## Ideas for later....

(1) Pragmatic categorisation
In a real world scanario, one would want to use all data available to do reliable categorisation, even if that would result in a 'less pretty' pipeline of classifiers including simple rules. The initial exploration of category labels shows that there is a number of sources which seem to publish articles from 1 category only, ie. specialised sources. Fig2 shows that 40% of sources serve only 1 category (when considering sources with more than 10 articles in the dataset). When we only consider sources with more than 50 articles in the dataset, the picture becomes more bimodal, sources are either heterogenuous with respect to the categories they serve (9-12 categories) or they only cover a few (1-4) categories. Together these more 'specialised' (serving 1-4 categories) sources still make up 30% of all sources.
Therfore in a pragmatic approach, the source of the article (derived from the URL field here) could be used to assign the category to the portion of articles from 'specialised' sources before even looking at the text itself. Alternatively, one could use a Baysian approach that allows to set a prior for category probability depening on the source.

(2) Developing a prior for each source/outlet
Similar to the Baysion approach mentioned above, one could think of learning a background model for the text itself, not for the entirety of the corpus but with respect to the source the article comes from. Every source will have its own specific vocabulary when writing about science, culture, politics etc respectively. Developing a classifier that would distinguish these categoeries not across but source-specific could make use of some distincive vocabulary that would get lost in the 'noise' of many sources. However, for this approach to work one would probably need more labeled data to develop reliable priors.




