import random

from pandas import read_csv

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


class Zoo:
    """This class encapsulates the Friedman1 test for a regressor
    """

    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    NUM_FOLDS = 5

    def __init__(self, randomSeed):
        """
        :param randomSeed: random seed value used for reproducible results
        """
        self.randomSeed = randomSeed

        # read the dataset, skipping the first columns (animal name):
        self.data = read_csv(self.DATASET_URL, header=None, usecols=range(1, 18))

        # separate to input features and resulting category (last column):
        self.X = self.data.iloc[:, 0:16]
        self.y = self.data.iloc[:, 16]

        # split the data, creating a group of training/validation sets to be used in the k-fold validation process:
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

        self.classifier = DecisionTreeClassifier(random_state=self.randomSeed)

    def __len__(self):
        """
        :return: the total number of features used in this classification problem
        """
        return self.X.shape[1]

    def getMeanAccuracy(self, zeroOneList):
        """
        returns the mean accuracy measure of the calssifier, calculated using k-fold validation process,
        using the features selected by the zeroOneList
        :param zeroOneList: a list of binary values corresponding the features in the dataset. A value of '1'
        represents selecting the corresponding feature, while a value of '0' means that the feature is dropped.
        :return: the mean accuracy measure of the calssifier when using the features selected by the zeroOneList
        """

        # drop the dataset columns that correspond to the unselected features:
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndices], axis=1)

        # perform k-fold validation and determine the accuracy measure of the classifier:
        cv_results = model_selection.cross_val_score(self.classifier, currentX, self.y, cv=self.kfold, scoring='accuracy')

        # return mean accuracy:
        return cv_results.mean()


# testing the class:
def main():
    # create a problem instance:
    zoo = Zoo(randomSeed=42)

    allOnes = [1] * len(zoo)
    print("-- All features selected: ", allOnes, ", accuracy = ", zoo.getMeanAccuracy(allOnes))


if __name__ == "__main__":
    main()
