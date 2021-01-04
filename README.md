# Upday Challenge


I advocate for multidisciplinary colaboration with non IT professionals, so I like prepare solutions that are readily available using no more than a web browser.

For this challenge, I prepared a quick React sketch that shows an interface where an user can input text to be classified into the 12 different caregories. The frontend is served by a Django backend, which has a copy of a trained classifier model and can be used through an API. 

I hosted the solution on my personal GKE, which can be accesed at staging.huentelemu.com/upday-challenge. For this, I had to create my own private repo for security reasons. I can send you invitation if you want to check it out.

Please use your favorite package manager to install the Python required libraries from Pipfile or requirements.txt files, for example:

```
pipenv install -r requirements.txt
```

Please also make sure you have Spacy's pretrained model for English. In the case of using pipenv, the command would be:

```
pipenv run python -m spacy download en_core_web_lg
```

After this, you can run:
```
pipenv run jupyter-lab
```

Once the research notebook is thoroughly ran, the model will be saved on disk. From here on you can mount the solution locally using:

```
docker-compose up
```

This fetches a Python image with the relevant NLP libraries installed, and Spacy trained models downloaded. Once it's ready, you can access the solution in localhost:8000.
