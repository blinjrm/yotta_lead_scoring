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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.domain.cleaning as cleaning
import src.infrastructure.make_dataset as infra
import src.settings.base as stg
from src.application.model import create_stacked_model
from src.domain.build_features import (FeatureSelector, NumericalTransformer,
                                       add_durre_moy_par_visite,
                                       add_nb_visites_null, drop_indexes,
                                       drop_quality_niveau_lead, drop_scores,
                                       regroupe_category_origine,
                                       regroupe_create_category_autre)

stg.enable_logging(log_filename=f'{basename(__file__)}.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the raw data')
filename = PARSER.parse_args().filename

logging.info('_'*39)
logging.info('_________ Launch new analysis _________\n')

df0 = infra.DatasetBuilder(filename).data

df_before = cleaning.DataCleaner(filename=filename).entry_data
df_clean = cleaning.DataCleaner(filename=filename).clean_data

X = df_clean.drop(columns=stg.TARGET)
y = df_clean[stg.TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

num_pipeline = make_pipeline(FeatureSelector(np.number),
                             add_nb_visites_null(),
                             add_durre_moy_par_visite(),
                             drop_scores(),
                             #NumericalTransformer(),
                             # Deal with outliers - transformation log
                             SimpleImputer(strategy='median'),
                             StandardScaler()
                             )

cat_pipeline = make_pipeline(FeatureSelector('category'),
                             drop_quality_niveau_lead(),
                             drop_indexes(),
                             regroupe_category_origine(),
                             regroupe_create_category_autre(),
                             SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore")
                             )

data_pipeline = make_union(num_pipeline, cat_pipeline)

if not os.path.exists(os.path.join(stg.MODEL_DIR, 'stacked_model.pkl')):
    X_train_transformed = data_pipeline.fit_transform(X_train, y_train)
    X_valid_transformed = data_pipeline.transform(X_test)

    logging.info('finding best hyperparameters...\n')
    create_stacked_model(X_train_transformed, y_train, X_valid_transformed, y_test)

logging.info('loading model from model/...\n')
filename = os.path.join(stg.MODEL_DIR, 'stacked_model.pkl')
with open(filename, 'rb') as f:
    stacked_model = pickle.load(f)

full_pipeline = make_pipeline(data_pipeline, stacked_model)

logging.info('Training model...\n')
full_pipeline.fit(X_train, y_train)

# Renvoie 0 ou 1
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('\naccuracy : ', accuracy)

# Renvoie la probabilité d'être converti
y_pred = full_pipeline.predict_proba(X_test)[:,1]
log_loss = log_loss(y_test, y_pred)
print('\nlog_loss : ', log_loss, '\n')

data_with_prediction = X_test.copy()
data_with_prediction['prediction'] = pd.Series(y_pred, index=data_with_prediction.index)                                      
data_with_prediction = data_with_prediction.sort_values(by=['prediction'], ascending=False)
