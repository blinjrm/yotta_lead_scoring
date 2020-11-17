"""This module prepares the dataset to use in the model.

Classes
-------
FeatureSelector
NumericalTransformer
CategoricalTransformer

"""


from sklearn.base import BaseEstimator, TransformerMixin
import src.settings.base as stg
import logging


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


'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import src.settings.base as stg
import src.settings.column_names as column_names
import src.infrastructure.make_dataset as make_dataset
import src.domain.cleaning as cleaning

from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.model_selection import train_test_split


filename = "data.csv"
data = cleaning.DataCleaner(filename).clean_data

X = data.drop(columns=stg.TARGET)
y = data[stg.TARGET].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train2 = add_nb_visites_null().transform(X_train)
X_train3 = add_durre_moy_par_visite().transform(X_train2)
X_train4 = drop_scores().transform(X_train3)
X_train5 = drop_indexes().transform(X_train4)
X_train6 = drop_quality_niveau_lead().transform(X_train5)
X_train7 = regroupe_category_origine().transform(X_train6)
inst = regroupe_create_category_autre().fit(X_train7)
X_train8 = inst.transform(X_train7)

X_test2 = add_nb_visites_null().transform(X_test)
X_test3 = add_durre_moy_par_visite().transform(X_test2)
X_test4 = drop_scores().transform(X_test3)
X_test5 = drop_indexes().transform(X_test4)
X_test6 = drop_quality_niveau_lead().transform(X_test5)
X_test7 = regroupe_category_origine().transform(X_test6)
X_test8 = inst.transform(X_test7)

print("data columns", X_train8.columns,'\n')
print("\n",X_train8.head().T)
print("train",X_train8[stg.ORIGINE_LEAD_COL].value_counts(dropna=False),'\n')
print("test",X_test8[stg.ORIGINE_LEAD_COL].value_counts(dropna=False),'\n')

my_pipeline = make_pipeline(add_nb_visites_null(),
                            add_durre_moy_par_visite(),
                            drop_scores(),
                            drop_indexes(),
                            drop_quality_niveau_lead(),
                            regroupe_category_origine(),
                            regroupe_create_category_autre()
                             )


X = data.drop(columns=stg.TARGET)
y = data[stg.TARGET].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorial_col = X_train.select_dtypes(include='category').columns

inst = my_pipeline.fit(X_train)
X_train2 = inst.transform(X_train)
X_test2 = inst.transform(X_test)

categorial_col = X_train2.select_dtypes(include='category').columns

for f in categorial_col :
    
        print("\ntrain",f,"\n",X_train[f].value_counts(dropna=False))
        print("\ntrain after",f,"\n",X_train2[f].value_counts(dropna=False))
        print("\ntest after",f,"\n",X_test2[f].value_counts(dropna=False))

'''

