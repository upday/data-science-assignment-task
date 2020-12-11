## Requirements and set up
1. you must have [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-mac/) installed on your machine 
2. FIRST TIME ONLY: run `docker-compose build` to build the docker image where the code run ran.

## Instructions
1. run `make train` to train the model (or if it doesn't work, `docker-compose run --rm notebook /bin/bash -c "python3 xgboost_train.py"`)
2. run `make evaluate DATASET="<path of your dataset>"` to evaluate the model (alternatively `docker-compose run --rm notebook /bin/bash -c "python3 model_evaluate.py '<path of your dataset>'"`). Replace [path of your dataset] with your dataset path (NOTE: must be the same format as the one provided)

To explore and modify the jupyter notebook where I explore and compare different models:
1. `docker-compose up`
2. copy and paste on fo the URLs on your browser
3. select `exploration.ipynb`