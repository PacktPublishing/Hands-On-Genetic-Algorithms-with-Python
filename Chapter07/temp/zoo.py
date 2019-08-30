import numpy as np

from sklearn import model_selection
from sklearn import datasets

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from pandas import read_csv

class Zoo:
    """This class encapsulates the Friedman1 test for a regressor
    """

    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    VALIDATION_SIZE = 0.20
    NOISE = 1.0
    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.data = read_csv(self.DATASET_URL, header=None, usecols=range(1, 18))

        self.X = self.data.iloc[:, 0:16]
        self.y = self.data.iloc[:, 16]

        self.X_train, self.X_validation, self.y_train, self.y_validation = \
            model_selection.train_test_split(self.X, self.y, test_size=self.VALIDATION_SIZE, random_state=self.randomSeed)

        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

        self.classifier = DecisionTreeClassifier(random_state=self.randomSeed)
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

    def getAccuracyScore1(self, zeroOneList):

        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX_train = self.X_train.drop(self.X_train.columns[zeroIndices], axis=1)
        currentX_validation = self.X_validation.drop(self.X_validation.columns[zeroIndices], axis=1)

        self.classifier.fit(currentX_train, self.y_train)
        prediction = self.classifier.predict(currentX_validation)

        return accuracy_score(self.y_validation, prediction)

    def getAccuracyScore(self, zeroOneList):
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndices], axis=1)

        cv_results = model_selection.cross_val_score(self.classifier, currentX, self.y, cv=self.kfold, scoring='accuracy')

        return cv_results.mean()


# testing the class:
def main():
    # create a problem instance:
    zoo = Zoo(42)

    scores = []
    # calculate MSE for 'n' first features:
    for n in range(1, len(zoo) + 1):
        nFirstFeatures = [1] * n + [0] * (len(zoo) - n)
        score = zoo.getAccuracyScore(nFirstFeatures)
        print("%d first features: score = %f" % (n, score))
        scores.append(score)

    # plot graph:
    plt.plot([i + 1 for i in range(len(test))], scores, color='red')
    plt.xticks(np.arange(1, len(zoo) + 1, 1.0))
    plt.xlabel('n First Features')
    plt.ylabel('MSE')
    plt.title('MSE vs. Features Selected')
    plt.show()


if __name__ == "__main__":
    main()
