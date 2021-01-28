FROM python:3.9
COPY src/ src
COPY data/ data
RUN pip install --upgrade pip
RUN pip install -r src/requirements.txt
RUN pip3 install jupyter
WORKDIR src
CMD sh

