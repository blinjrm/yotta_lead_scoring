"""This module prepares the dataset to use in the model.

Classes
-------
NumericalTransformer
CategoricalTransformer
BooleanTransformer

"""


from sklearn.base import BaseEstimator, TransformerMixin

import logging


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


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """Transforms the categorical columns.

    Attributes
    ----------
    none

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
    #     X_with_country = self._tranform_country(X)
    #     X_with_city = self._transform_city(X_with_country)
    #     X_without_nan = self._replace_all_missing_values(X_with_city)
    #     return X_without_nan.values
        X = X.fillna('Non_renseigne')
        return X

    # def _tranform_country(self, X):
    #     X.loc[(X['PAYS'] != 'India') | (X['PAYS'] != 'United States'), 'PAYS'] = 'Autre'
    #     return X

    # def _transform_city(self, X):
    #     X.loc[X['VILLE'].isna(), 'VILLE'] = 'Non_renseigné'
    #     return X

    # def _replace_all_missing_values(self, X):
    #     X = X.fillna('Non_renseigné')
    #     return X


class BooleanTransformer(BaseEstimator, TransformerMixin):
    """Transforms the categorical columns.

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
