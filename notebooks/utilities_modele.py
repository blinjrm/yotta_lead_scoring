

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, plot_precision_recall_curve


def reporting_greadsearch_graph(X_train, y_train, X_test,y_test, data_pipeline, param_grid, scoring, class_weight=None):

    rf = RandomForestClassifier(random_state=42, class_weight=class_weight)

    full_pipeline = make_pipeline(data_pipeline, rf)
    gs = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, cv=5, scoring=scoring)
    gs.fit(X_train, y_train)
    y_pred = gs.predict(X_test)

    max_depth = gs.best_params_['randomforestclassifier__max_depth']
    min_samples_split = min_samples_split = gs.best_params_['randomforestclassifier__min_samples_split']
    nb_estimators = gs.best_params_['randomforestclassifier__n_estimators']

    print(" Best parameters : nb_estimators = {}, max_depth = {}, min_samples_split = {}\n".format(nb_estimators, max_depth, min_samples_split) )

    # refit et faire une matrice de confuison et graph
    rf = RandomForestClassifier(random_state=42, max_depth = max_depth, min_samples_split = min_samples_split , n_estimators = nb_estimators, class_weight=class_weight )
    
    full_pipeline = make_pipeline(data_pipeline, rf)

    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    y_pred_proba = full_pipeline.predict_proba(X_test)

    from sklearn.metrics import plot_confusion_matrix

    print("Le Random Forest a pour 'accuracy' : %.3f." % accuracy_score(y_test, y_pred))
    print("Le Random Forest a pour 'precision' : %.3f." % precision_score(y_test, y_pred))
    print("Le Random Forest a pour 'recall' : %.3f." % recall_score(y_test, y_pred))
    print("Le Random Forest a pour 'f1' : %.3f." % f1_score(y_test, y_pred))
    print("\n")

    class_names = np.array(['non converti', 'converti'])

    disp = plot_confusion_matrix(full_pipeline, X_test, y_test,
                                    display_labels=class_names,
                                    cmap=plt.cm.Blues)

    disp.ax_.set_title("Confusion matrix")


    y_score = full_pipeline.predict_proba(X_test)
    precision, recall, threshold = precision_recall_curve(y_test, y_score[:, 1])

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')

    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Courbe de Précision-Rappel')

    return full_pipeline


def reporting_greadsearch(X_train, y_train, X_test,y_test,data_pipeline, param_grid, scoring, class_weight=None):

    rf = RandomForestClassifier(random_state=0, class_weight=class_weight)

    full_pipeline = make_pipeline(data_pipeline, rf)
    gs = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, cv=5, scoring=scoring)
    gs.fit(X_train, y_train)
    y_pred = gs.predict(X_test)

    max_depth = gs.best_params_['randomforestclassifier__max_depth']
    min_samples_split = min_samples_split = gs.best_params_['randomforestclassifier__min_samples_split']
    nb_estimators = gs.best_params_['randomforestclassifier__n_estimators']

    print(" Best parameters : nb_estimators = {}, max_depth = {}, min_samples_split = {}\n".format(nb_estimators, max_depth, min_samples_split) )

    # refit et faire une matrice de confuison et graph
    rf = RandomForestClassifier(random_state=0, max_depth = max_depth, min_samples_split = min_samples_split , n_estimators = nb_estimators, class_weight=class_weight )
    full_pipeline = make_pipeline(data_pipeline, rf)

    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    y_pred_proba = full_pipeline.predict_proba(X_test)

    from sklearn.metrics import plot_confusion_matrix

    print("Le Random Forest a pour 'accuracy' : %.3f." % accuracy_score(y_test, y_pred))
    print("Le Random Forest a pour 'precision' : %.3f." % precision_score(y_test, y_pred))
    print("Le Random Forest a pour 'recall' : %.3f." % recall_score(y_test, y_pred))
    print("Le Random Forest a pour 'f1' : %.3f." % f1_score(y_test, y_pred))
    print("\n")

    return full_pipeline
