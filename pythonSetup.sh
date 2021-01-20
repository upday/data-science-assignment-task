
#!/bin/bash


# nb how to start in higher directory
# jupyter notebook --notebook-dir ~/Work

# https://medium.com/dunder-data/anaconda-is-bloated-set-up-a-lean-robust-data-science-environment-with-miniconda-and-conda-forge-b48e1ac11646

# bash  ~/Downloads/Miniconda3-latest-Linux-x86_64.sh

echo 'PATH=$PATH:$HOME/miniconda3/bin' >> ~/.bashrc 
source ~/.bashrc

conda config --set auto_activate_base false

conda create -n basic_ds -y

conda activate basic_ds

# conda config --show channels

conda config --env --add channels conda-forge

# cat miniconda3/envs/basic_ds/.condarc

conda config --env --set channel_priority strict
conda config --show channel_priority

conda install pandas scikit-learn -y
conda install gensim -y

conda install pip git -y
conda install nltk -y
conda install scipy -y
conda install textblob -y
conda install langdetect -y
conda install xgboost -y
#conda install googletrans -y
pip install googletrans==4.0.0-rc1


conda install StringIO -y
conda install urllib2 -y


conda install -c conda-forge -y
