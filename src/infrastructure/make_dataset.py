"""This module creates and cleans the dataset from a flat file.

Classes
-------
DatasetBuilder
FeatureSelector

"""


import logging

import numpy as np
import pandas as pd

import src.settings.base as stg


class DatasetBuilder:
    """Creates dataet from CSV file.

    Attributes
    ----------
    data: dataset in a Pandas dataframe

    """

    def __init__(self, filename, mode):
        self.data = self._load_data_from_csv(filename, mode)

    def _load_data_from_csv(self, filename, mode):
        if self._check_file_extension(filename):
            df = self._open_file(filename, mode)
            return df

    def _check_file_extension(self, filename):
        logging.info('-'*20)
        logging.info('Confirm file extension is .csv ..')
        if filename.endswith('.csv'):
            logging.info('.. Done \n')
            return True
        else:
            logging.info('.. ERROR: Extension must be .csv')
            raise FileExistsError('Extension must be .csv')

    def _open_file(self, filename, mode):
        logging.info('-'*20)
        logging.info('Load data ..')
        if mode == 'train':
            DIR = stg.RAW_DATA_DIR
        else:
            DIR = stg.PREDICTION_DATA_DIR
        try:
            df = pd.read_csv(''.join((DIR, filename)), sep=';')
            logging.info('.. Done \n')
            return df
        except FileNotFoundError as error:
            logging.info('.. FileNotFoundError')
            raise FileNotFoundError(f'Error in SalesDataset initialization - {error}')


class DataCleaner:
    """Cleans the dataset:
    - remove accents
    - remove uppercase
    - drop columns

    Attributes
    ----------
    data: dataset in a Pandas dataframe

    """

    def __init__(self, df):
        self.data = self._clean_dataset(df)

    def _clean_dataset(self, df):
        df_without_accents = self._remove_accents(df)
        df_without_uppercase = self._remove_upper_case(df_without_accents)
        df_without_duplicates = df_without_uppercase.drop_duplicates()
        df_without_unknown_columns = self._remove_unknown_columns(df_without_duplicates)
        df_with_index = self._set_ID_as_index(df_without_unknown_columns)
        df_without_nan_target = self._drop_samples_without_target(df_with_index)
        df_with_category_type = self._change_object_type_to_category(df_without_nan_target)
        return df_with_category_type

    def _remove_accents(self, df):
        cols = df.select_dtypes(include=[np.object]).columns
        df[cols] = df[cols].apply(lambda x: x.str.normalize('NFKD')\
                           .str.encode('ascii', errors='ignore').str.decode('utf-8'))
        return df

    def _remove_upper_case(self, df):
        cols = df.select_dtypes(include=[np.object]).columns
        df[cols] = df[cols].apply(lambda x: x.str.lower())
        return df

    def _remove_unknown_columns(self, df): 
        for col in df.columns:
            if col not in stg.FEATURES + [stg.TARGET]:
                df = df.drop(columns=col)
        return df

    def _set_ID_as_index(self, df):
        df_with_index = df.set_index(stg.ID_CLIENT_COL)
        return df_with_index

    def _drop_samples_without_target(self, df):
        df = df.copy()
        df_without_target_nan = df.dropna(subset=[stg.CONVERTI_COL])
        return df_without_target_nan

    def _change_object_type_to_category(self, df):
        df_changed = df.copy()
        categorial_cols = df.select_dtypes(include=object).columns
        for col in categorial_cols:
            df_changed[col] = df_changed[col].astype("category")
        return df_changed
