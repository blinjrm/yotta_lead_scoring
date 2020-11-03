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
        if self._check_file_extension(filename):
            df = self._open_file(filename)
            df_without_accents = self._remove_accents(df)
            df_without_uppercase = self._remove_upper_case(df_without_accents)
            return df_without_uppercase

    def _check_file_extension(self, filename):
        logging.info('-'*20)
        logging.info('Confirm file extension is .csv ..')
        if filename.endswith('.csv'):
            logging.info('.. Done')
            return True
        else:
            logging.info('.. Extension must be .csv')
            raise FileExistsError('Extension must be .csv')

    def _open_file(self, filename):
        logging.info('-'*20)
        logging.info('Load data ..')
        try:
            df = pd.read_csv(''.join((stg.RAW_DATA_DIR, filename)), sep=';')
            logging.info('.. Done')
            return df
        except FileNotFoundError as error:
            logging.info('.. FileNotFoundError')
            raise FileNotFoundError(f'Error in SalesDataset initialization - {error}')

    def _remove_accents(self, df):
        cols = df.select_dtypes(include=[np.object]).columns
        df[cols] = df[cols].apply(lambda x: x.str.normalize('NFKD')\
                           .str.encode('ascii', errors='ignore').str.decode('utf-8'))
        return df

    def _remove_upper_case(self, df):
        cols = df.select_dtypes(include=[np.object]).columns
        df[cols] = df[cols].apply(lambda x: x.str.lower())
        return df

    @property
    def split_features(self):
        pass
        # numerical_features = np numeric


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Filters dataset using the selected features 
    (numerical vs. categorical vs. boolean)
    """
    
    def __init__(self, _dtype):
        self._dtype = _dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.select_dtypes(include=self._dtype)
