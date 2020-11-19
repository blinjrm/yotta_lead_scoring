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


stg.enable_logging(log_filename='project_logs.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the data to make predictions')
filename = PARSER.parse_args().filename

logging.info('_'*42)
logging.info('_________ Launch new prediction __________\n')


    # Deal with the different directory !!!

# X_predict = infra.DatasetBuilder(filename).data
X_predict = cleaning.DataCleaner(filename=filename).clean_data

if stg.TARGET in X_predict.columns:
    X_predict.drop(columns=stg.TARGET, inplace=True)


try:
    logging.info('Loading existing model from model/..')
    with open(stg.SAVED_MODEL_FILE, 'rb') as f:
        full_pipeline = pickle.load(f)
    logging.info('.. Done \n')
except FileNotFoundError:
    logging.info('.. Error: no trained model has been found in model/')
    raise


logging.info('Using model for predictions..')
y_predict = full_pipeline.predict_proba(X_predict)[:,1]
logging.info('.. Done \n')


data_with_prediction = X_predict.copy()
data_with_prediction['prediction'] = pd.Series(y_predict, index=data_with_prediction.index)                                      
data_with_prediction = data_with_prediction.sort_values(by=['prediction'], ascending=False)
