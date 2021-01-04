import os
from urllib.request import urlretrieve
from zipfile import ZipFile

def download_data(data_folder='data'):

    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    url = 'https://upday-data-assignment.s3-eu-west-1.amazonaws.com/science/data.zip'
    urlretrieve(url, data_folder + '/data.zip')

    print('Please enter password for Zip file:')
    pwd = str.encode(input())

    with ZipFile('data/data.zip') as zf:
        zf.extractall(path=data_folder + '/', pwd=pwd)
