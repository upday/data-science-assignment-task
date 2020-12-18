# Upday Task for Data Scientists

## Introduction

The repo contains the analysis notebook, scripts for training, testing models, API server wrapper for the model, Docker file to run notebooks and API. All of my data analysis and model training/evaluation is in the notebooks. The readme will help you run the scripts as smoothly aspossible.

## Prerequisites

You need to have python3 running on your system and install pipenv

```
pip3 install pipenv
```

## Notebooks

There are two ways to run notebooks:

### Directly

Install required developement dependencies
```
pipenv sync -d --clear
```
Run the notebook
```
jupyter notebook
```
Remember to copy the data file url and password to notebook before running.

### Docker

You need to have docker installed. Build the container and run it with the command
```
docker build -f Dockerfile.notebook -t notebook:0.1 .
docker run -p 8888:8888 notebook:0.1
```
You can access notebook server from localhost:8888

## Model Training and Evaluation Scripts

To run the scripts, you need to install required dependencies
```
pipenv sync -d --clear
```
All the configuration, path, variable as well as url and password to data file is stored `config\application.yaml` file. You can leave the default parameters, except data zipfile url and password.

### Download and unzip data file
This will download the zipfile and unzip to the default location defined in application.yaml. Make sure you have correct url and password in the yaml file.
```
pipenv run python model/download_data.py
pipenv run python model/download_data.py --url <url-of-data> --password <password>
```
### Split to train and test dataset
The script will read the whole dataset and do a stratified split into train and test dataset. The parameters can be set in application.yaml files or passed as parameters
```
pipenv run python model/split_train_test.py
pipenv run python model/split_train_test.py --input <path-to-input-file> --train <path-to-output-train-dataset> -test <path-to-output-test-dataset> --size <test-size>
```
### Train model
The script will train a fasttext model based on train dataset. By default it will use the whole data file, you can specify it to use a another file instead (after splitting for example)
```
pipenv run python model/train_model.py
pipenv run python model/train_model.py --input <path-to-train-file> --output <path-to-model-file>
```
### Evaluate
The script loads the model and calculates the Precision, Recall and F1 Score based on a test dataset. Please make sure to provide the correct path to test dataset or it will use the default location for splitted test dataset.
```
pipenv run python model/evaluate_model.py --test <path-to-dataset>
```
### Predict 
The script loads the model and will try to predict the category of an article you provided. You need to input the article url, title and text.
```
pipenv run python model/predict_model.py --url "http://ww.edheu.cd" --title "new life in health and style" --text "health and style are very important"
```

## API Wrapper
The API is created by FastAPI library and provide very basic endpoint for other services. FastAPI has good performance and support async requests.

### Preparation
Please note that you need to train a model before running the API. Put the data url and password inside application.yaml then run following commands
```
pipenv sync --clear
pipenv run python model/download_data.py
pipenv run python model/train_model.py
```
### Run API server
To run the API:

```
pipenv run python server/main.py
```

Or docker

```
docker build -f Dockerfile.api -t api:0.1 .
docker run -p 8080:8080 api:0.1
```
### How To Use
You can see a simple Swagger docs for the API at `http://0.0.0.0:8080/docs`
The API has one endpoint `/predict_label` accepting JSON request with article information. Here is one example request
```
curl --header "Content-Type: application/json" --data '{
    "url": "www.test.com",
    "title": "new life in health and style",
    "text": "health and style are very important"}
' http://0.0.0.0:8080/predict_label
```
Response
```
{"label":"fashion_beauty_lifestyle","probability":1.0000100135803223}
```