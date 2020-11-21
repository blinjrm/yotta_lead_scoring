"""
Contains all configurations for the project.
Should NOT contain any secrets.

>>> import src.settings as stg
>>> stg.COL_NAME
"""

import logging
import os

from src.settings.column_names import *

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW_DATA_DIR = os.path.join(REPO_DIR, 'data/training/')
PREDICTION_DATA_DIR = os.path.join(REPO_DIR, 'data/prediction/')
OUTPUTS_DIR = os.path.join(REPO_DIR, 'outputs')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')
MODEL_DIR = os.path.join(REPO_DIR, 'model')


def enable_logging(log_filename, logging_level=logging.DEBUG):
    """Set loggings parameters.

    Parameters
    ----------
    log_filename: str
    logging_level: logging.level

    """
    with open(os.path.join(LOGS_DIR, log_filename), 'a') as file:
        file.write('\n')
        file.write('\n')

    LOGGING_FORMAT = '[%(asctime)s][%(levelname)s][%(module)s] - %(message)s'
    LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        level=logging_level,
        filename=os.path.join(LOGS_DIR, log_filename)
    )


SAVED_MODEL_FILE = os.path.join(MODEL_DIR, 'model.pkl')

OUTLIER_ID = 602958

CATEGORY_MIN_THRESHOLD = 10

CONSTANT_FEATURES_TO_DROP = [
    MAGAZINE_COL,
    SOUHAITE_RECEVOIR_INFOS_COL,
    SOUHAITE_RECEVOIR_MAJ_COL,
    SOUHAITE_RECEVOIR_MSG_COL,
    SOUHAITE_PAYER_CHEQUE_COL,
]

OTHER_FEATURES_TO_DROP = [
    NB_PAGES_VUES_PAR_VISITE_COL,
    DERNIERE_ACTIVITE_NOTABLE_COL,
    TAGS_COL
]

FEATURES_WITH_LOW_SECOND_CATEGORY_TO_DROP = [
    CONTACT_PAR_TELEPHONE_COL,
    ANNONCE_VUE_COL,
    ARTICLE_JOURNAL_COL,
    FORUM_COL,
    JOURNAUX_COL,
    PUB_DIGITALE_COL,
    RECOMMANDATION_COL
]

FEATURES_NUM_TO_DROP = [
    SCORE_ACTIVITE_COL,
    SCORE_PROFIL_COL
]

FEATURES_CAT_TO_DROP = [
    INDEX_ACTIVITE_COL,
    INDEX_PROFIL_COL,
    QUALITE_LEAD_COL,
    NIVEAU_LEAD_COL,
    PAYS_COL
]
