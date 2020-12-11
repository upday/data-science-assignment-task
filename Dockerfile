FROM python:3.7.9-slim-buster

COPY ./requirements.txt /requirements/
RUN apt-get update && apt-get install -y libgomp1
RUN pip3 install --upgrade pip
RUN pip3 -vv install --no-cache-dir -r /requirements/requirements.txt

WORKDIR /var/app
COPY . . 
CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root