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
from warnings import simplefilter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

import src.settings.base as stg
from src.application.model import create_pipeline
from src.domain.build_features import AddFeatures

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

stg.enable_logging(log_filename='project_logs.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the raw data')
filename = PARSER.parse_args().filename


logging.info('_'*20)
logging.info('_________ Launch new training ___________\n')


df = AddFeatures(filename=filename, mode='train').data_with_all_features

X = df.drop(columns=stg.TARGET)
y = df[stg.TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
data_pipeline = create_pipeline()
full_pipeline = make_pipeline(data_pipeline, rf)

param_grid = {'randomforestclassifier__n_estimators': [30, 50, 70],
              'randomforestclassifier__min_samples_split': [5, 10],
              'randomforestclassifier__max_depth': [3, 5, 7, 9]}

logging.info('Finding best hyperparameters for new model..')
clf = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, cv=5)

logging.info('Training model..')
clf.fit(X_train, y_train)

logging.info('Evaluating model..')
y_pred = clf.predict(X_test)

print(f'\nThe model was successfully trained, with a precision of {precision_score(y_test, y_pred)}%.')
print("classification_report\n", classification_report(y_test, y_pred))

logging.info('Saving trained model..')
with open(stg.SAVED_MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)
logging.info('Model successfully trained.')
