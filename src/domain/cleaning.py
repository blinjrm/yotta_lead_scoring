"""This module does some additional cleaning based on the conclusions of the data analysis

Classes
-------
DataCleaner

"""


import numpy as np
import pandas as pd

import src.infrastructure.make_dataset as infra
import src.settings.base as stg


class DataCleaner:
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

    def __init__(self, filename, mode):
        """Initialize class.

        Parameters
        ----------
        filename: str
            CSV filename containing data

        """

        self.entry_data = infra.DataCleaner(infra.DatasetBuilder(filename, mode).data).data

    @property
    def clean_data(self):
        """Main methode to clean the data"""
        df = self.entry_data.copy()
        df_with_cleaning = self._clean(df)
        return df_with_cleaning

    def _clean(self, df):
        df = df.copy()
        df_without_non_exploitable_features = self._drop_not_exploitable_features(df)
        df_without_constants = self._remove_constants(df_without_non_exploitable_features)
        df_without_features_low_second_category = self._drop_features_with_low_second_category(df_without_constants)
        df_without_outliers_errors = self._correct_outliers_errors(df_without_features_low_second_category)
        df_with_corrected_niveau_lead = self._correct_select_niveau_lead(df_without_outliers_errors)
        df_with_category_formulaire_add = self._add_category_formulaire_add(df_with_corrected_niveau_lead)
        df_with_category_etudiant = self._group_to_category_etudiant(df_with_category_formulaire_add)
        df_with_category_en_activite = self._create_category_en_activite(df_with_category_etudiant)
        df_with_change_derniere_activite = self._regroup_categories_derniere_activite(df_with_category_en_activite)
        df_with_corrected_pays = self._correct_unknow_pays_to_nan(df_with_change_derniere_activite)
        return df_with_corrected_pays

    @staticmethod
    def _drop_not_exploitable_features(df):
        df = df.copy()
        df_with_drop_features = df.drop(stg.OTHER_FEATURES_TO_DROP, axis=1)
        return df_with_drop_features

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

        mask_outlier = (df[stg.NB_VISITES_COL] != 0) & (df[stg.DUREE_SUR_SITEWEB_COL] == 0)
        df[stg.NB_VISITES_COL] = np.where(mask_outlier, np.nan, df[stg.NB_VISITES_COL])
        df[stg.DUREE_SUR_SITEWEB_COL] = np.where(mask_outlier,
                                                 np.nan,
                                                 df[stg.DUREE_SUR_SITEWEB_COL]
                                                 )
        df[stg.NB_VISITES_COL] = np.where((df[stg.NB_VISITES_COL] == 251) & (df.index == stg.OUTLIER_ID),
                                          np.nan,
                                          df[stg.NB_VISITES_COL]
                                          )
        df[stg.DUREE_SUR_SITEWEB_COL] = np.where((df[stg.DUREE_SUR_SITEWEB_COL] == 49.0) & (df.index == stg.OUTLIER_ID),
                                                 np.nan,
                                                 df[stg.DUREE_SUR_SITEWEB_COL]
                                                 )
        return df

    @staticmethod
    def _correct_select_niveau_lead(df):
        df = df.copy()
        df[stg.NIVEAU_LEAD_COL] = df[stg.NIVEAU_LEAD_COL].replace('select', np.nan)
        return df

    @staticmethod
    def _add_category_formulaire_add(df):
        df = df.copy()
        df[stg.ORIGINE_LEAD_COL] = df[stg.ORIGINE_LEAD_COL].replace("formulaire quick add", "formulaire add")
        df[stg.ORIGINE_LEAD_COL] = df[stg.ORIGINE_LEAD_COL].replace("formulaire lead add", "formulaire add")
        return df

    @staticmethod
    def _group_to_category_etudiant(df):
        df = df.copy()
        df[stg.NIVEAU_LEAD_COL] = df[stg.NIVEAU_LEAD_COL].replace("etudiant d'une certaine ecole", "etudiant")
        df[stg.NIVEAU_LEAD_COL] = df[stg.NIVEAU_LEAD_COL].replace("etudiant en double specialisation", "etudiant")
        return df

    @staticmethod
    def _create_category_en_activite(df):
        df = df.copy()
        df[stg.STATUT_ACTUEL_COL] = df[stg.STATUT_ACTUEL_COL].replace("homme d'affaire", "en activite")
        df[stg.STATUT_ACTUEL_COL] = df[stg.STATUT_ACTUEL_COL].replace("professionnel en activite", "en activite")
        return df

    @staticmethod
    def _regroup_categories_derniere_activite(df):
        df = df.copy()
        df[stg.DERNIERE_ACTIVITE_COL] = df[stg.DERNIERE_ACTIVITE_COL].replace("reinscrit aux emails", "formulaire soumis sur le site")
        df[stg.DERNIERE_ACTIVITE_COL] = df[stg.DERNIERE_ACTIVITE_COL].replace("stand visite au salon", "approche directe")
        df[stg.DERNIERE_ACTIVITE_COL] = df[stg.DERNIERE_ACTIVITE_COL].replace("email marque comme spam", "ne veut pas de contact")
        df[stg.DERNIERE_ACTIVITE_COL] = df[stg.DERNIERE_ACTIVITE_COL].replace("desinscrit", "ne veut pas de contact")
        df[stg.DERNIERE_ACTIVITE_COL] = df[stg.DERNIERE_ACTIVITE_COL].replace("email rejete", "ne veut pas de contact")
        df[stg.DERNIERE_ACTIVITE_COL] = df[stg.DERNIERE_ACTIVITE_COL].replace("a clique sur le lien dans le navigateur", "a clique sur le lien")
        df[stg.DERNIERE_ACTIVITE_COL] = df[stg.DERNIERE_ACTIVITE_COL].replace("a clique sur le lien dans le mail", "a clique sur le lien")
        df[stg.DERNIERE_ACTIVITE_COL] = df[stg.DERNIERE_ACTIVITE_COL].replace("a clique sur le lien dand le navigateur", "a clique sur le lien")
        return df

    @staticmethod
    def _correct_unknow_pays_to_nan(df):
        df[stg.PAYS_COL] = df[stg.PAYS_COL].replace("unknown", np.nan)
        return df
