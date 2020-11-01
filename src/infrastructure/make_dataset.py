"""This module creates and cleans the dataset from a flat file.

Classes
-------
DatasetBuilder
FeatureSelector

"""


from sklearn.base import BaseEstimator, TransformerMixin

import logging
import numpy as np
import pandas as pd

import src.settings.base as stg


class DatasetBuilder:
    """Creates dataet from flat file.

    Attributes
    ----------
    data: dataset in a Pandas dataframe

    """

    def __init__(self, filename):
        self.data = self._load_data_from_csv(filename)

    def _load_data_from_csv(self, filename):
        logging.info('-'*20)
        logging.info('Confirm file extension is .csv ..')
        if filename.endswith('.csv'):
            logging.info('.. Done')
            logging.info('-'*20)
            logging.info('Load data ..')
            try:
                df = pd.read_csv(''.join((stg.RAW_DATA_DIR, filename)), sep=';')
                logging.info('.. Done')
            except FileNotFoundError as error:
                logging.info('.. FileNotFoundError')
                raise FileNotFoundError(f'Error in SalesDataset initialization - {error}')
        else:
            logging.info('.. Extension must be .csv')
            raise FileExistsError('Extension must be .csv')
        df_without_accents = self._remove_accents(df)
        return df_without_accents

    def _remove_accents(self, df):
        cols = df.select_dtypes(include=[np.object]).columns
        df[cols] = df[cols].apply(lambda x: x.str.normalize('NFKD')\
                           .str.encode('ascii', errors='ignore').str.decode('utf-8'))
        return df


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Filters dataset using the selected features 
    (numerical vs. categorical vs. boolean)
    """
    
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]
