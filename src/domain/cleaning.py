"""This module does some additional cleaning based on the conclusions of the data analysis

Classes
-------
DataCleaner

"""


import pandas as pd
import numpy as np
import logging

import src.infrastructure.make_dataset as infra 
import src.settings.base as stg


class DataCleaner :

    """ Add some additional cleaning based on the conclusions of the data analysis

    Attributes
    ----------
    entry_data: pandas.DataFrame
        Dataframe including the dataset before the additional cleaning

    Properties
    ----------
    clean_data: pandas.DataFrame
        DataFrame obtained after the cleaning

    """


    def __init__(self, filename):

        """Initialize class.

        Parameters
        ----------
        filename: str
            CSV filename containing data

        """

        self.entry_data = infra.DataCleaner(infra.DatasetBuilder(filename).data).data

    @property
    def clean_data(self):
        """Main methode to clean the data"""
        df = self.entry_data.copy()
        df_with_cleaning = self._clean(df)
        return df_with_cleaning
    

    def _clean(self,df):
        df = df.copy()
        df_with_category_type = self._change_object_type_to_category(df)
        df_without_non_exploitable_features = self._drop_not_exploitable_features(df_with_category_type)
        df_without_constants = self._remove_constants(df_without_non_exploitable_features)
        df_without_features_low_second_category = self._drop_features_with_low_second_category(df_without_constants)
        df_without_outliers_errors = self._correct_outliers_errors(df_without_features_low_second_category)
        return  df_without_outliers_errors

    @staticmethod
    def _drop_not_exploitable_features(df):
        df = df.copy()
        df_with_drop_features = df.drop(stg.OTHER_FEATURES_TO_DROP,axis=1)
        return df_with_drop_features
    
    @staticmethod
    def _change_object_type_to_category(df):
        df_changed = df.copy()
        numerical_cols = df.select_dtypes(include=np.number).columns
        categorial_cols = df.select_dtypes(include=object).columns
        for col in categorial_cols :
            df_changed[col] = df_changed[col].astype("category")
        return df_changed

    @staticmethod
    def _drop_features_with_low_second_category(df):
        df = df.copy()
        df_with_drop_features = df.drop(stg.FEATURES_WITH_LOW_SECOND_CATEGORY_TO_DROP, axis=1)
        return df_with_drop_features
    
    @staticmethod
    def _remove_constants(df):
        df = df.copy()
        df_without_constants = df.drop(stg.CONSTANT_FEATURES_TO_DROP, axis=1)
        return df_without_constants

    @staticmethod
    def _correct_outliers_errors(df):
        df = df.copy()
    
        mask_outlier = (df[stg.NB_VISITES_COL] != 0) & (df[stg.DUREE_SUR_SITEWEB_COL]==0) 
        df[stg.NB_VISITES_COL] = np.where(mask_outlier, np.nan, df[stg.NB_VISITES_COL])
        df[stg.DUREE_SUR_SITEWEB_COL] = np.where(mask_outlier, np.nan, df[stg.DUREE_SUR_SITEWEB_COL])

        df[stg.NB_VISITES_COL] = np.where((df[stg.NB_VISITES_COL]==251) & (df.index==stg.OUTLIER_ID), np.nan, df[stg.NB_VISITES_COL])
        df[stg.DUREE_SUR_SITEWEB_COL] = np.where((df[stg.DUREE_SUR_SITEWEB_COL]==49.0) & (df.index==stg.OUTLIER_ID), np.nan, df[stg.DUREE_SUR_SITEWEB_COL])
    
        return df    




  






    