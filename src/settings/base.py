"""
Contains all configurations for the project.
Should NOT contain any secrets.

>>> import src.settings as stg
>>> stg.COL_NAME
"""

import os
import yaml
import logging

import src.settings.column_names as column_names


# By default the raw data is stored in this repository's "data/raw/" folder.
# You can change it in your own settings file.
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW_DATA_DIR = os.path.join(REPO_DIR, 'data/raw/')
OUTPUTS_DIR = os.path.join(REPO_DIR, 'outputs')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')


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


FEATURES = column_names.FEATURES
LABEL = column_names.LABEL
