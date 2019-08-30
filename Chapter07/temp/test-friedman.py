# -*- coding: utf-8 -*-
"""
Configuring Python environment for machine learning basics
"""
# loading libraries

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import datasets
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from pprint import pprint

NUM_FEATURES = 15
RANDOM_SEED = 42
VALIDATION_SIZE = 0.20
NUM_SAMPLES = 60
NOISE = 0.0
NUM_FOLDS = 3

#--CFH: use num featues - 10, but remove the last 5 features to see the difference

#for num_features in range(5, 25):
    # creating the dataset
X_orig, y = datasets.make_friedman1(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, noise=NOISE, random_state=RANDOM_SEED)

#for i in range(1, (NUM_FEATURES - 5) + 1):
for i in range(0, NUM_FEATURES):

    #X = X_orig.drop(X_orig.iloc[-i], inplace = False, axis = 1)
    X = X_orig[:, :-i] if i > 0 else X_orig


    X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)

    regressor = GradientBoostingRegressor()
    #regressor = RandomForestRegressor()
    #regressor = ExtraTreesRegressor(n_estimators=10)
    #regressor = LinearRegression()

    regressor.fit(X_train, y_train)
    prediction = regressor.predict(X_validation)
    mse = mean_squared_error(y_validation, prediction)
    #pprint(list(zip(y_validation, prediction)))
    #print("mse = ", mse)

    scoring = 'neg_mean_squared_error'
    kfold = model_selection.KFold(n_splits=NUM_FOLDS, random_state=RANDOM_SEED)

    #loo = model_selection.LeaveOneOut()

    #cv_results = model_selection.cross_val_score(regressor, X_train, y_train, cv=kfold, scoring=scoring)

    cv_results = model_selection.cross_val_score(regressor, X, y, cv=kfold, scoring=scoring)
    #cv_results = model_selection.cross_val_score(regressor, X, y, cv=loo, scoring=scoring)

    #msg = "%d: %s %f (%f)" % (num_features, "Mean, Std = ", cv_results.mean(), cv_results.std())
    msg = "%d, %d: mse = %f mean = %f" % (i, NUM_FEATURES - i, mse, cv_results.mean())
    print(msg)

