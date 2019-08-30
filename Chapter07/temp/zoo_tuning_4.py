import numpy as np

from sklearn import model_selection
from sklearn import datasets

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from pandas import read_csv

class ZooTuning:
    """This class encapsulates the Friedman1 test for a regressor
    """

    #DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

    VALIDATION_SIZE = 0.20
    NOISE = 1.0
    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        #self.data = read_csv(self.DATASET_URL, header=None, usecols=range(1, 18))
        self.data = read_csv(self.DATASET_URL, header=None, usecols=range(0, 14))

        #self.X = self.data.iloc[0:16]
        #self.y = self.data.iloc[:, 16]
        self.X = self.data.iloc[:, 1:14]
        self.y = self.data.iloc[:, 0]

        self.X_train, self.X_validation, self.y_train, self.y_validation = \
            model_selection.train_test_split(self.X, self.y, test_size=self.VALIDATION_SIZE, random_state=self.randomSeed)

        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

        self.classifier = DecisionTreeClassifier(random_state=self.randomSeed)
        #self.classifier = SVC(random_state=self.randomSeed, gamma='scale')
        #self.classifier = AdaBoostClassifier(random_state=self.randomSeed)

        #self.classifier = GradientBoostingClassifier(random_state=self.randomSeed)
        #self.classifier = LogisticRegression(random_state=self.randomSeed, multi_class='auto', solver='lbfgs')
        #self.classifier = KNeighborsClassifier()
        #self.classifier = GaussianNB()


    def __len__(self):
        """
        :return: the total number of items defined in the problem
        """
        return self.X.shape[1]


    def getAccuracyScore(self, params):

        max_depth_value = None if round(params[0]) == 0 else round(params[0])
        criterion_value = ['gini', 'entropy'][round(params[1])]
        splitter_value = ['best', 'random'][round(params[2])]

        self.classifier = DecisionTreeClassifier(random_state=self.randomSeed, max_depth=max_depth_value, criterion=criterion_value, splitter=splitter_value)

        cv_results = model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.kfold, scoring='accuracy')

        return cv_results.mean()


# testing the class:
def main():
    # create a problem instance:
    zoo = ZooTuning(42)

    print("Classifier: ", zoo.classifier.get_params())
    print("score (1.0, 'scale') = ", zoo.getAccuracyScore([1.0, 'scale']))
    print("score (1.0, 'auto') = ", zoo.getAccuracyScore([1.0, 'auto']))
    print("optimized score = ", zoo.getAccuracyScore([1.9, 0.236]))


if __name__ == "__main__":
    main()
