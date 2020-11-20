"""Module to train the model.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/train.py -f data.csv

Script could be run with the following command line from a python interpreter :

    >>> run src/application/train.py -f data.csv

Attributes
----------
PARSER: argparse.ArgumentParser

"""

import argparse
import logging
import os
import pickle
from os.path import basename, join

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# import src.domain.cleaning as cleaning
# import src.infrastructure.make_dataset as infra
import src.settings.base as stg
from src.application.model import create_model, create_pipeline
from src.domain.build_features import AddFeatures

stg.enable_logging(log_filename='project_logs.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the raw data')
PARSER.add_argument('--stacked', '-s', default=False, action='store_true', help='True to use a stacked model, default False')
filename = PARSER.parse_args().filename
stacked_model = PARSER.parse_args().stacked


logging.info('_'*20)
logging.info('_________ Launch new training ___________\n')


df = AddFeatures(filename=filename).data_with_all_fetaures

X = df.drop(columns=stg.TARGET)
y = df[stg.TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

data_pipeline = create_pipeline()

logging.info('Finding best hyperparameters for new model..')
X_train_transformed = data_pipeline.fit_transform(X_train, y_train)
X_valid_transformed = data_pipeline.transform(X_test)

model = create_model(X_train_transformed, y_train, X_valid_transformed, y_test, stacked_model)
logging.info('.. Done \n')

pipeline = make_pipeline(data_pipeline, model)

logging.info('Training model..')
pipeline.fit(X_train, y_train)
logging.info('.. Done \n')


logging.info('Saving trained model..')
with open(stg.SAVED_MODEL_FILE, 'wb') as f:
    pickle.dump(pipeline, f)
logging.info('.. Done \n')

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nThe model was successfully trained, with an accuracy of {accuracy}%.')

print("precision_recall_curve\n", precision_recall_curve(y_test, y_pred))

print("classification_report\n", classification_report(y_test, y_pred))
