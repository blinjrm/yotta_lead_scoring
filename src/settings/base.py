"""
Contains all configurations for the project.
Should NOT contain any secrets.

>>> import src.settings as stg
>>> stg.COL_NAME
"""

import os
import logging

# By default the raw data is stored in this repository's "data/raw/" folder.
# You can change it in your own settings file.
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW_DATA_DIR = os.path.join(REPO_DIR, 'data/raw/')
OUTPUTS_DIR = os.path.join(REPO_DIR, 'outputs')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')


# Logging
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


# Categrical columns
categorical_features = ['DERNIERE_ACTIVITE', 
                        'DERNIERE_ACTIVITE_NOTABLE', 
                        'PAYS', 
                        'VILLE', 
                        'ORIGINE_LEAD', 
                        'SOURCE_LEAD', 
                        'SPECIALISATION', 
                        'STATUT_ACTUEL', 
                        'Comment avez-vous entendu parler de nous ?']

# Numerical columns
numerical_features = ['NB_VISITES', 
                      'DUREE_SUR_SITEWEB', 
                      'NB_PAGES_VUES_PAR_VISITE']

# Boolean columns
boolean_features = ['CONTACT_PAR_MAIL', 
                    'Souhaites-tu recevoir une copie de notre livre blanc ?']
