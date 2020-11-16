"""This module prepares the dataset to use in the model.

Classes
-------
FeatureSelector
NumericalTransformer
CategoricalTransformer

"""


from sklearn.base import BaseEstimator, TransformerMixin

import logging


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



class NumericalTransformer(BaseEstimator, TransformerMixin):
    """Transforms the numerical columns.

    Attributes
    ----------
    none

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.values



class NewFeatures():
    """Define new features based on categorial features

    Attributes
    ----------
    none

    """

    def __init__(self):
        pass

    def add_nb_visites_null (self,df):
        # appliquer après imputation
        df = df.copy()
        df['NB_VISITES_IS_NULL'] = df['NB_VISITES'].apply(lambda x: 1 if x==0 else 0)
        return df

    def add_duree_moy_par_visite (self,df):
        # appliquer après imputation
        df = df.copy()
        df['NB_VISITES_IS_NULL'] = df['NB_VISITES'].apply(lambda x: 1 if x==0 else 0)
        return df
    
    def regroupe_pays(self,df):
        df = df.copy()
        return df



