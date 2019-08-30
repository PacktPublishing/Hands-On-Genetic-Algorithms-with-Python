# -*- coding: utf-8 -*-
"""
Configuring Python environment for machine learning basics
"""
# loading libraries

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# loading the dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

##shape
#
# print(dataset.shape)
#
##preview of the dataset
#
# print(dataset.head(20))
#
##descriptions, statistical summary
#
# print(dataset.describe())
#
##count(groupby(class))
#
# print(dataset.groupby('class').size())
#
##univariate plot to understand each of the attributes
#
# dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False, sharey=False)
# plt.show()
#
##histogram representation of the dataset
#
# dataset.hist()
# plt.show()
#
##multivariate plots
#
# scatter_matrix(dataset)
##plt.show()

# creating a training and cross validation set

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
A = array[:, 0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# ten-fold cross validation

num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

# evaluating different algorithms

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# comparing algorithms

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# predictions on cross validation set

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))