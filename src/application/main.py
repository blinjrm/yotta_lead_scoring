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
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_recall_curve, classification_report
from sklearn.preprocessing import FunctionTransformer

import argparse
import logging
import numpy as np
import pandas as pd

import src.domain.cleaning as cleaning  
from src.domain.build_features import FeatureSelector
import src.settings.base as stg
from src.domain.build_features import DropIndexes, DropQualityAndNiveauLead, DropScores
from src.domain.build_features import RegroupeCreateCategoryAutre
from src.domain.build_features import AddFeatures
import src.infrastructure.make_dataset as infra 



stg.enable_logging(log_filename=f'{basename(__file__)}.log', logging_level=logging.INFO)

PARSER = argparse.ArgumentParser(description='File containing the dataset.')
PARSER.add_argument('--filename', '-f', required=True, help='Name of the file containing the raw data')
filename = PARSER.parse_args().filename

logging.info('_'*39)
logging.info('_________ Launch new analysis _________\n')


df = AddFeatures(filename="data.csv").data_with_all_fetaures

X = df.drop(columns=stg.TARGET)
y = df[stg.TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
# TODO essayer d'adopter pour recuperer les noms des colonnes
def get_column_names_from_ColumnTransformer(column_transformer):    
    col_name = []
    for transformer_in_columns in column_transformer.transformers_:#the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1],Pipeline): 
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError: # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names,np.ndarray): # eg.
            col_name += names.tolist()
        elif isinstance(names,list):
            col_name += names    
        elif isinstance(names,str):
            col_name.append(names)
    return col_name
'''


num_pipeline = make_pipeline(FeatureSelector(np.number),
                            DropScores(),
                            FunctionTransformer(np.log1p),
                            # winsorization 
                            SimpleImputer(strategy='median'),
                            RobustScaler(),
                            #StandardScaler()
                             )

cat_pipeline = make_pipeline(FeatureSelector('category'),
                            DropQualityAndNiveauLead(),
                            DropIndexes(),
                            RegroupeCreateCategoryAutre(),
                            SimpleImputer(strategy="most_frequent"),
                            # frequency encoder
                            OneHotEncoder(handle_unknown="ignore")
                             )

data_pipeline = make_union(num_pipeline, cat_pipeline)

#full_pipeline = make_pipeline(data_pipeline, LogisticRegression(max_iter=1000))
full_pipeline = make_pipeline(data_pipeline, RandomForestClassifier(class_weight='balanced'))

full_pipeline.fit(X_train, y_train)

# Renvoie 0 ou 1
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('\naccuracy : ', accuracy)

#precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
#print('\nprecision : ', precision, '\n')

print("precision_recall_curve\n", precision_recall_curve(y_test, y_pred))

print("classification_report\n", classification_report(y_test, y_pred))

# Renvoie la probabilité d'être converti
y_pred = full_pipeline.predict_proba(X_test)[:,1]
log_loss = log_loss(y_test, y_pred)
print('\nlog_loss : ', log_loss, '\n')

data_with_prediction = X_test.copy()
data_with_prediction['prediction'] = pd.Series(y_pred, index=data_with_prediction.index)                                      
data_with_prediction = data_with_prediction.sort_values(by=['prediction'], ascending=False)
#print(data_with_prediction)
