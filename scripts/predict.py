"""
Usage:
    predict.py <ARTICLE>

This functions expects an article file to be passed to the predictor.
That file should just be a plain text file with the title on the first line.
"""
from docopt import docopt
from upday.data_loader import ModelSaver
from upday import ml

args = docopt(__doc__)
filename = args['<ARTICLE>']

model = ModelSaver().load()

with open(filename) as f:
    title = f.readline()
    text = f.read()

category = ml.predict([title], [text], model)

print(f'Article in file {filename} is about {category[0]}')
