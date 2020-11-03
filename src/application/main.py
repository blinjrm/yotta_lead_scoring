"""Module to build dataset.

Example
-------
Script could be run with the following command line from the shell :

    $ python src/application/main.py -f data.csv

Script could be run with the following command line from a python interpreter :

    >>> run src/application/main.py -f data.csv

Attributes
----------
PARSER: argparse.ArgumentParser

"""

from os.path import basename, join
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

import argparse
import logging
import numpy as np
import pandas as pd

import src.infrastructure.make_dataset as make_dataset
import src.domain.build_features as build_features
import src.settings.base as stg


stg.enable_logging(log_filename=f'{basename(__file__)}.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the raw data')
filename = PARSER.parse_args().filename

logging.info('_'*39)
logging.info('_________ Launch new analysis _________\n')

DT = make_dataset.DatasetBuilder(filename=filename).data

X = DT.drop(stg.LABEL, axis=1)
y = DT[stg.LABEL].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # test_size should be 0.2


num_pipeline = make_pipeline(make_dataset.FeatureSelector(np.number),
                             build_features.NumericalTransformer(),
                             SimpleImputer(strategy='median'),
                             StandardScaler())

cat_pipeline = make_pipeline(make_dataset.FeatureSelector(object),
                             build_features.CategoricalTransformer(),
                             OneHotEncoder(handle_unknown="ignore"))

data_pipeline = make_union(num_pipeline, cat_pipeline)

full_pipeline = make_pipeline(data_pipeline, LogisticRegression(max_iter=1000))
# full_pipeline = make_pipeline(data_pipeline, RandomForestClassifier())

full_pipeline.fit(X_train, y_train)

# Renvoie 0 ou 1
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('\naccuracy : ', accuracy)

# Renvoie la probabilité d'être converti
y_pred = full_pipeline.predict_proba(X_test)[:,1]
log_loss = log_loss(y_test, y_pred)
print('\nlog_loss : ', log_loss, '\n')

data_with_prediction = X_test
data_with_prediction['prediction'] = pd.Series(y_pred, index=data_with_prediction.index)                                      
data_with_prediction = data_with_prediction.sort_values(by=['prediction'], ascending=False)
print(data_with_prediction)
