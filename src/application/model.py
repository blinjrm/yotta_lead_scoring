''' 


'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingCVClassifier

import os
import optuna
import pickle
import src.settings.base as stg


def objective_RF(trial, X_train, y_train, X_valid, y_valid):

    param = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 20),
        'max_depth': int(trial.suggest_float('max_depth', 1, 32, log=True))
    }

    rf = RandomForestClassifier(**param)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy


def objective_CatB(trial, X_train, y_train, X_valid, y_valid):

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
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy


def objective_SVC(trial, X_train, y_train, X_valid, y_valid):

    param = {
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf']),
        'C': trial.suggest_float('C', 0.01, 0.1),
        'gamma': trial.suggest_int('gamma', 1, 10)
    }

    svc = SVC(**param)
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy


def tune_hyperparameters(X_train, y_train, X_valid, y_valid):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_RF = optuna.create_study(direction="maximize")
    study_RF.optimize(lambda trial: objective_RF(trial, X_train, y_train, X_valid, y_valid), n_trials=100)
    rf = RandomForestClassifier(**study_RF.best_params)
    # rf.fit(X_train, y_train)

    study_CatB = optuna.create_study(direction="maximize")
    study_CatB.optimize(lambda trial: objective_CatB(trial, X_train, y_train, X_valid, y_valid), n_trials=100)
    catb = CatBoostClassifier(**study_CatB.best_params, verbose=0)
    # catb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100)

    # study_SVC = optuna.create_study(direction="maximize")
    # study_SVC.optimize(lambda trial: objective_SVC(trial, X_train, y_train, X_valid, y_valid), n_trials=100)
    # svc = SVC(**study_SVC.best_params, probability=True)
    # svc.fit(X_train, y_train)

    return rf, catb #, svc


def create_stacked_model(X_train, y_train, X_valid, y_valid):
    rf, catb = tune_hyperparameters(X_train, y_train, X_valid, y_valid)
    lr = LogisticRegression()

    sclf = StackingCVClassifier(classifiers=[rf, catb],
                                use_probas=True,
                                meta_classifier=lr,
                                random_state=42
                                )

    sclf.fit(X_train, y_train)

    filename = os.path.join(stg.MODEL_DIR, 'stacked_model.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(sclf, f)


if __name__ == "__main__":

    import pandas as pd
    import src.domain.cleaning as cleaning
    import category_encoders as ce
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    df_clean = cleaning.DataCleaner(filename='data.csv').clean_data

    X = df_clean.drop(columns=stg.TARGET)
    y = df_clean[stg.TARGET].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    data_pipeline = make_pipeline(ce.TargetEncoder(),
                                  SimpleImputer(strategy='median'),
                                  MinMaxScaler()
                                  )

    X_train = data_pipeline.fit_transform(X_train, y_train)
    X_valid = data_pipeline.transform(X_valid)
