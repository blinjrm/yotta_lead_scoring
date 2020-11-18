"""This module prepares the dataset to use in the model.

Classes
-------
FeatureSelector
NumericalTransformer
CategoricalTransformer

"""


import logging

from sklearn.base import BaseEstimator, TransformerMixin

import src.settings.base as stg


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Filters dataset using the selected features
    (numerical vs. categorical)
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




class add_nb_visites_null(BaseEstimator, TransformerMixin):
    """add the boolean feature "number of visites is null"
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = X.copy()
        X['NB_VISITES_IS_NULL']= X['NB_VISITES']
        X.loc[X['NB_VISITES_IS_NULL']>0,['NB_VISITES_IS_NULL']]=-1
        X.loc[X['NB_VISITES_IS_NULL']==0,['NB_VISITES_IS_NULL']]=0
        X.loc[X['NB_VISITES_IS_NULL']==-1,['NB_VISITES_IS_NULL']]=1

        return X


class add_durre_moy_par_visite(BaseEstimator, TransformerMixin):
    """add the feature DUREE_MOY_PAR_VISITE
    """

    def __init__(self):
        pass

    def fit(self, X,y=None):
        return self

    def transform(self, X,y=None):
        
        x = X.copy()
        X['NB_DUREE_MOY_PAR_VISITE']= X['NB_VISITES']
        X.loc[X['NB_VISITES']>0,['NB_DUREE_MOY_PAR_VISITE']]=X['DUREE_SUR_SITEWEB']/X['NB_VISITES']
        X.loc[X['NB_VISITES']==0,['NB_DUREE_MOY_PAR_VISITE']]=0
        X = X.drop(['DUREE_SUR_SITEWEB'],axis=1)

        return X



class drop_scores(BaseEstimator, TransformerMixin):
    """drop SCORE_ACTIVITE and SCORE_PROFILE
    """

    def __init__(self):
        pass

    def fit(self, X,y=None):
        return self

    def transform(self, X,y=None):
        X = X.copy()
        X = X.drop([stg.SCORE_ACTIVITE_COL,stg.SCORE_PROFIL_COL],axis=1)
        return X


class drop_indexes(BaseEstimator, TransformerMixin):
    """drop INDEX_PROFIL and INDEX_ACTIVITE
    """

    def __init__(self):
        pass

    def fit(self, X,y=None):
        return self

    def transform(self, X,y=None):
        X = X.copy()
        X = X.drop([stg.INDEX_ACTIVITE_COL,stg.INDEX_PROFIL_COL],axis=1)
        return X


class drop_quality_niveau_lead(BaseEstimator, TransformerMixin):
    """drop INDEX_PROFIL and INDEX_ACTIVITE
    """

    def __init__(self):
        pass

    def fit(self, X,y=None):
        return self

    def transform(self, X,y=None):
        X = X.copy()
        X = X.drop([stg.QUALITE_LEAD_COL,stg.NIVEAU_LEAD_COL],axis=1)
        return X


class regroupe_category_origine(BaseEstimator, TransformerMixin):
    """regroup categories "formulaire quick add" and "formulaire lead add" to "formulaire add"
    """

    def __init__(self):
        pass

    def fit(self, X,y=None):
        return self

    def transform(self, X,y=None):
        X = X.copy()
        X[stg.ORIGINE_LEAD_COL] = X[stg.ORIGINE_LEAD_COL].replace( "formulaire quick add","formulaire add")
        X[stg.ORIGINE_LEAD_COL] = X[stg.ORIGINE_LEAD_COL].replace( "formulaire lead add","formulaire add",)
        return X


class regroupe_create_category_autre(BaseEstimator, TransformerMixin):
    """regroup categories with less than categor_min_threshold and create the category "Autre" """

    def __init__(self):
        self.mapping = []
        pass

    def fit(self, X,y=None):
        categorial_col = X.select_dtypes(include='category').columns
        print(categorial_col)

        for col in categorial_col :
            counts = X[col].value_counts(dropna=False)
            mapping_list = list(counts[counts>stg.CATEGORY_MIN_THRESHOLD].index.dropna())
            self.mapping.append(mapping_list)

        return self


    def transform(self, X,y=None):

        X = X.copy()
        categorial_col = X.select_dtypes(include='category').columns

        i=0
        for col in categorial_col :

            temp = X[col].apply(lambda x: x if x in self.mapping[i] else 'autre')
            X[col] = temp 
            i=i+1
        
        return X

