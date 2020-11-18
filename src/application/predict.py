"""Module to make predictions on a dataset.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/predict.py -f data.csv

Script could be run with the following command line from a python interpreter :

    >>> run src/application/predict.py -f data.csv

Attributes
----------
PARSER: argparse.ArgumentParser

"""


import argparse
import errno
import logging
import os
import pickle
import sys
from os.path import basename, join

import pandas as pd

import src.domain.cleaning as cleaning
import src.infrastructure.make_dataset as infra
import src.settings.base as stg


stg.enable_logging(log_filename=f'{basename(__file__)}.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the data to make predictions')
filename = PARSER.parse_args().filename

logging.info('_'*42)
logging.info('_________ Launch new prediction _________\n')


    # Deal with the different directory !!!

# X_predict = infra.DatasetBuilder(filename).data
X_predict = cleaning.DataCleaner(filename=filename).clean_data

model_filename = os.path.join(stg.MODEL_DIR, 'stacked_model.pkl')
try:
    with open(model_filename, 'rb') as f:
        stacked_model = pickle.load(f)
        sys.exit()
except FileNotFoundError:
    raise

y_predict = stacked_model.predict_proba(X_predict)[:,1]

data_with_prediction = X_predict.copy()
data_with_prediction['prediction'] = pd.Series(y_predict, index=data_with_prediction.index)                                      
data_with_prediction = data_with_prediction.sort_values(by=['prediction'], ascending=False)
