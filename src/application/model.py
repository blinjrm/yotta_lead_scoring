"""Module to create the pipeline and define the optimal classifier
using hyperparameter optimization and model stacking
"""


import logging

import category_encoders as ce
import numpy as np
import optuna
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

import src.settings.base as stg
from src.domain.build_features import (DropFeatures, FeatureSelector,
                                       RegroupeCreateCategoryAutre)


def create_pipeline():
    """ create the pipeline preparing the data to be fed into the classifier
    """
    num_pipeline = make_pipeline(FeatureSelector(np.number),
                                 DropFeatures(stg.FEATURES_NUM_TO_DROP),
                                 FunctionTransformer(np.log1p),
                                 SimpleImputer(strategy='median', add_indicator=True),
                                 )

    cat_pipeline = make_pipeline(FeatureSelector('category'),
                                 DropFeatures(stg.FEATURES_CAT_TO_DROP),
                                 RegroupeCreateCategoryAutre(),
                                 SimpleImputer(strategy="most_frequent", add_indicator=True),
                                 ce.TargetEncoder()
                                 )

    data_pipeline = make_union(num_pipeline, cat_pipeline)

    return data_pipeline


def objective_RF(trial, X_train, y_train, X_valid, y_valid):
    """
    define the Optuna objective function to find
    the optimal hyperparameters for the random forest model.
    """

    param = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 20),
        'max_depth': int(trial.suggest_float('max_depth', 1, 32, log=True)),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 10)
    }

    rf = RandomForestClassifier(**param)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_valid)
    result = precision_score(y_valid, y_pred, zero_division=0)

    return result


def objective_CatB(trial, X_train, y_train, X_valid, y_valid):
    """
    define the Optuna objective function to find
    the optimal hyperparameters for the CatBoost model.
    """

    param = {
        'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 1, 12),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
    }
    if param['bootstrap_type'] == 'Bayesian':
        param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif param['bootstrap_type'] == 'Bernoulli':
        param['subsample'] = trial.suggest_float('subsample', 0.1, 1)

    catb = CatBoostClassifier(**param)
    catb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0, early_stopping_rounds=30)

    y_pred = catb.predict(X_valid)
    result = precision_score(y_valid, y_pred, zero_division=0)

    return result


def tune_random_forest(X_train, y_train, X_valid, y_valid):
    """
    Define Optuna studies and optimize the hyperparameters
    of the models.
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_RF = optuna.create_study(direction="maximize")
    study_RF.optimize(lambda trial: objective_RF(trial, X_train, y_train, X_valid, y_valid), n_trials=100)
    rf = RandomForestClassifier(**study_RF.best_params)

    return rf


def tune_catboost(X_train, y_train, X_valid, y_valid):
    """
    Define Optuna studies and optimize the hyperparameters
    of the models.
    """

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_CatB = optuna.create_study(direction="maximize")
    study_CatB.optimize(lambda trial: objective_CatB(trial, X_train, y_train, X_valid, y_valid), n_trials=100)
    catb = CatBoostClassifier(**study_CatB.best_params, verbose=0)

    return catb


def create_model(X_train, y_train, X_valid, y_valid, stacked_model):
    """create a  model.
    If stacked_model is true, create a model by combining the optimized random forest and catboost models
    with a logistic regression meta-classifier.
    Otherwise create a random forest model.
    """

    if stacked_model:
        logging.info('-> option : stacked model')
        print('Creating an optimized stacked model, this may take a while')

        rf = tune_random_forest(X_train, y_train, X_valid, y_valid)
        catb = tune_catboost(X_train, y_train, X_valid, y_valid)
        lr = LogisticRegression()

        model = StackingCVClassifier(classifiers=[rf, catb],
                                     use_probas=True,
                                     meta_classifier=lr,
                                     random_state=42
                                     )

    else:
        logging.info('-> option : random forest classifier')

        model = tune_random_forest(X_train, y_train, X_valid, y_valid)

    return model
