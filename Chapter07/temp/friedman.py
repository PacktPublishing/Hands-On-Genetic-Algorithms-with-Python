import numpy as np

from sklearn import model_selection
from sklearn import datasets

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class Friedman1Test:
    """This class encapsulates the Friedman1 test for a regressor
    """

    VALIDATION_SIZE = 0.20
    NOISE = 1.0

    def __init__(self, numFeatures, numSamples, randomSeed):

        self.numFeatures = numFeatures
        self.numSamples = numSamples

        # generate test data:
        self.randomSeed = randomSeed
        self.X, self.y = datasets.make_friedman1(n_samples=self.numSamples, n_features=self.numFeatures,
                                                 noise=self.NOISE, random_state=self.randomSeed)

        self.X_train, self.X_validation, self.y_train, self.y_validation = \
            model_selection.train_test_split(self.X, self.y, test_size=self.VALIDATION_SIZE, random_state=self.randomSeed)

        self.regressor = GradientBoostingRegressor(random_state=self.randomSeed)

    def __len__(self):
        """
        :return: the total number of items defined in the problem
        """
        return self.numFeatures

    #TODO: remove
    def getKfoldMeanError(self, zeroOneList):

        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = np.delete(self.X, zeroIndices, 1)

        kFold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)
        cvResults = model_selection.cross_val_score(self.regressor, currentX, self.y, cv=kFold, scoring='neg_mean_squared_error')

        return cvResults.mean()

    def getMeanError(self, zeroOneList):

        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX_train = np.delete(self.X_train, zeroIndices, 1)
        currentX_validation = np.delete(self.X_validation, zeroIndices, 1)

        self.regressor.fit(currentX_train, self.y_train)
        prediction = self.regressor.predict(currentX_validation)

        return mean_squared_error(self.y_validation, prediction)


# testing the class:
def main():
    # create a problem instance:
    test = Friedman1Test(15, 60, 42)

    scores = []
    # calculate MSE for 'n' first features:
    for n in range(1, len(test) + 1):
        nFirstFeatures = [1] * n + [0] * (len(test) - n)
        score = test.getMeanError(nFirstFeatures)
        print("%d first features: score = %f" % (n, score))
        scores.append(score)

    # plot graph:
    plt.plot([i + 1 for i in range(len(test))], scores, color='red')
    plt.xticks(np.arange(1, len(test) + 1, 1.0))
    plt.xlabel('n First Features')
    plt.ylabel('MSE')
    plt.title('MSE vs. Features Selected')
    plt.show()


if __name__ == "__main__":
    main()
