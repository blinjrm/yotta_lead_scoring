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

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool

import src.domain.cleaning as cleaning
import src.infrastructure.make_dataset as infra
import src.settings.base as stg
from src.application.model import create_stacked_model, tune_random_forest, tune_catboost
from src.domain.build_features import (FeatureSelector, NumericalTransformer,
                                       add_durre_moy_par_visite,
                                       add_nb_visites_null, drop_indexes,
                                       drop_quality_niveau_lead, drop_scores,
                                       regroupe_category_origine,
                                       regroupe_create_category_autre)

stg.enable_logging(log_filename='project_logs.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the training data')
filename = PARSER.parse_args().filename

logging.info('_'*42)
logging.info('_________ Launch new training ___________\n')


df_clean = cleaning.DataCleaner(filename=filename).clean_data

X = df_clean.drop(columns=stg.TARGET)
y = df_clean[stg.TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

num_pipeline = make_pipeline(FeatureSelector(np.number),
                             add_nb_visites_null(),
                             add_durre_moy_par_visite(),
                             drop_scores(),
                             SimpleImputer(strategy='median'),
                             StandardScaler()
                             )

cat_pipeline = make_pipeline(FeatureSelector('category'),
                             drop_quality_niveau_lead(),
                             drop_indexes(),
                             regroupe_category_origine(),
                             regroupe_create_category_autre(),
                            #  ce.TargetEncoder(),
                             SimpleImputer(strategy="most_frequent")
                             )

data_pipeline = make_union(num_pipeline, cat_pipeline)


logging.info('Finding best hyperparameters for new model..')
X_train_transformed = data_pipeline.fit_transform(X_train, y_train)
X_valid_transformed = data_pipeline.transform(X_test)
# model = tune_random_forest(X_train_transformed, y_train, X_valid_transformed, y_test)
model = create_stacked_model(X_train_transformed, y_train, X_valid_transformed, y_test)

logging.info('.. Done \n')

full_pipeline = make_pipeline(data_pipeline, model)

logging.info('Training model..')
full_pipeline.fit(X_train, y_train)
logging.info('.. Done \n')

logging.info('Saving trained model..')
with open(stg.SAVED_MODEL_FILE, 'wb') as f:
    pickle.dump(full_pipeline, f)
logging.info('.. Done \n')

y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nThe model was successfully trained, with an accuracy of {accuracy}%.')
