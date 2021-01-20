# You can speed this up by removing the large word embedding enwiki_201880420_100d.txt if you make sure it is not set to use it.

# thought about each of these... not sure..
# FROM continuumio/miniconda3
FROM frolvlad/alpine-miniconda3

# very basic image, no WORKDIR etc..


COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "upday_ds", "/bin/bash", "-c"]


# RUN echo "Make sure pandas is installed:"
# RUN python -c "import pandas"

COPY data/ /data/
#COPY run.py .
#COPY predictor.py . -> for optParse predictor, only defaults exposed

COPY funs.py .
COPY flaskPredictor.py .
#ENTRYPOINT ["conda", "run", "-n", "upday_ds", "python", "run.py"]

#EXPOSE 5000
# 0.0.0.0:5000->5000/tcp   distracted_gagarin


ENTRYPOINT ["conda", "run", "-n", "upday_ds", "python", "-u", "flaskPredictor.py"]
