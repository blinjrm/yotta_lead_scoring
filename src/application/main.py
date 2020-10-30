"""
à rédiger

"""

from os.path import basename, join

import logging
import pandas as pd

import src.settings.base as stg


stg.enable_logging(log_filename=f'{basename(__file__)}.log', logging_level=logging.DEBUG)

logging.info('-----------------------')
logging.info('Load data ..')
try:
    df = pd.read_csv(f'{stg.RAW_DATA_DIR}data.csv', sep=';')
except FileNotFoundError as error:
    raise FileNotFoundError(f'Error in SalesDataset initialization - {error}')
logging.info('.. Successfully loaded date.')


