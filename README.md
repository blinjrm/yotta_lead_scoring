# db-jb

Projet N°1 - Advanced ML
Sujet : Lead Scoring for Yotta Executive Education

Copyright : 2020, Dorota Bailly & Jerome Blin

==============================

## Project Organization
------------

    ├── README.md                      <- The top-level README for developers using this project.
    │
    ├── activate.sh                    <- Sript to configure the environment (PYTHONPATH, dependencies, virtual environment). 
    │
    ├── data
    │   ├── processed                  <- The final, canonical data sets for modeling.
    │   └── raw                        <- The original, immutable data dump.
    │
    ├── logs                           <- Folder containing the logs
    │
    ├── model                          <- Folder containing the trained model, to be used for predictions. 
    │
    ├── notebooks                      <- Jupyter notebooks.
    │
    ├── outputs                        <- Folder contraining the predictions of the model.
    │
    ├── poetry.lock                    <- Lock file to secure the version of dependencies.
    │
    ├── pyproject.toml                 <- Poetry file with dependencies.
    │
    └── src                            <- Source code for use in this project.
        ├── __init__.py                <- Makes src a Python module.
        │
        ├── application                <- Scripts to train models and then use trained models to make predictions.
        │   ├── model.py
        │   ├── predict.py
        │   └── train.py
        │
        ├── domain                     <- Sripts to clean the data and include feature engineering
        │   ├── build_features.py
        │   └── cleaning.py
        │
        ├── infrastructure             <- Scripts to load the raw data in a Pandas DataFrame
        │   └── make_dataset.py
        │
        └── settings                   <- Scripts containing the settings
            ├── base.py
            └── column_names.py

==============================

# Getting Started

## 1. Clone this repository

```
$ git clone <this project>
$ cd <this project>
```

## 1. Setup your environment

Goal : 
- add the directory to the PYTHONPATH
- install the dependencies (if some are missing)
- create a local virtual environment in the folder `./.venv/` (if it does not exist already)
- activate the virtual environment

- First: check your python3 version:

    ```
    $ python3 --version
    # examples of outputs:
    Python 3.6.2 :: Anaconda, Inc.
    Python 3.7.2

    $ which python3
    /Users/benjamin/anaconda3/bin/python3
    /usr/bin/python3
    ```

    - If you don't have python3 and you are working on your mac: install it from [python.org](https://www.python.org/downloads/)
    - If you don't have python3 and are working on an ubuntu-like system: install from package manager:

        ```
        $ apt-get update
        $ apt-get -y install python3 python3-pip python3-venv
        ```

- Now that python3 is installed create and configure your environment:

    `$ source activate.sh`

    You sould **always** activate your environment when working on the project.


## 2. Train the model

- The raw data to train the model on must be in the data/raw/ directory
- Run the training script using : 
from the shell :
    $ python src/application/train.py -f data.csv

from a python interpreter :
    >>> run src/application/train.py -f data.csv

- The trained model will be saved in the model/ directory as a pickle file. 


## 3. Use the model for predictions

- The data used for prediction must be in the data/raw/ directory
- Run the training script using : 
from the shell :
    $ python src/application/predict.py -f data.csv

from a python interpreter :
    >>> run src/application/predict.py -f data.csv

- The file containing the predictions will be saved in the outputs/ directory as a csv file. 
