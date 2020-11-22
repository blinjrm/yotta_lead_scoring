"""Module to make predictions on a dataset.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/predict.py -f new_data.csv

Script could be run with the following command line from a python interpreter :

    >>> run src/application/predict.py -f new_data.csv

Attributes
----------
PARSER: argparse.ArgumentParser

"""


import argparse
import logging
import pickle

import pandas as pd
from scipy.sparse import data

import src.settings.base as stg
from src.domain.build_features import AddFeatures


def load_pipeline():
    try:
        logging.info('Loading existing model from model/..')
        with open(stg.SAVED_MODEL_FILE, 'rb') as f:
            pipeline = pickle.load(f)
        logging.info('.. Done \n')
    except FileNotFoundError:
        logging.info('.. Error: no trained model has been found in model/')
        raise
    return pipeline


def promissing_lead(df):
    data_with_prediction['lead_prometteur'] = 0
    data_with_prediction.iloc[0, -1] = 1
    for i in range(len(data_with_prediction)):
        conv = data_with_prediction.iloc[0:i+1, -2].agg('mean')
        if conv > 0.8:
            data_with_prediction.iloc[i, -1] = 1
        else:
            break
    return df


stg.enable_logging(log_filename='project_logs.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the data to make predictions')
filename = PARSER.parse_args().filename

logging.info('_'*20)
logging.info('_________ Launch new prediction __________\n')

X_predict = AddFeatures(filename=filename, mode='predict').data_with_all_features

if stg.TARGET in X_predict.columns:
    X_predict.drop(columns=stg.TARGET, inplace=True)

pipeline = load_pipeline()

logging.info('Using model for predictions..')
y_predict = pipeline.predict_proba(X_predict)[:, 1]
logging.info('.. Done \n')

data_with_prediction = X_predict.copy()
data_with_prediction['probabilite_de_conversion'] = pd.Series(y_predict, index=data_with_prediction.index)
data_with_prediction = data_with_prediction.sort_values(by=['probabilite_de_conversion'], ascending=False)

data_with_promissing_lead = promissing_lead(data_with_prediction)

logging.info('Exporting data with results..')
data_with_prediction.to_csv('outputs/data_with_predictions.csv')
logging.info('.. Done \n')
