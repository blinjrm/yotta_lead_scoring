"""Module to train the model.

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

import src.domain.cleaning as cleaning  
from src.domain.build_features import FeatureSelector, NumericalTransformer
import src.settings.base as stg
from src.domain.build_features import add_durre_moy_par_visite, add_nb_visites_null
from src.domain.build_features import drop_indexes, drop_quality_niveau_lead, drop_scores
from src.domain.build_features import regroupe_category_origine, regroupe_create_category_autre
import src.infrastructure.make_dataset as infra 


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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


#full_pipeline = make_pipeline(data_pipeline, LogisticRegression(max_iter=1000))
full_pipeline = make_pipeline(data_pipeline, RandomForestClassifier())

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
#print(data_with_prediction)
