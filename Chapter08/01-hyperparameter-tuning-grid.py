import numpy as np
import time
import random

from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from pandas import read_csv
from evolutionary_search import EvolutionaryAlgorithmSearchCV


class HyperparameterTuningGrid:

    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initWineDataset()
        self.initClassifier()
        self.initKfold()
        self.initGridParams()

    def initWineDataset(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

        self.data = read_csv(url, header=None, usecols=range(0, 14))
        self.X = self.data.iloc[:, 1:14]
        self.y = self.data.iloc[:, 0]

    def initClassifier(self):
        self.classifier = AdaBoostClassifier(random_state=self.randomSeed)

    def initKfold(self):
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS,
                                           random_state=self.randomSeed)

    def initGridParams(self):
        self.gridParams = {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'learning_rate': np.logspace(-2, 0, num=10, base=10),
            'algorithm': ['SAMME', 'SAMME.R'],
        }

    def getDefaultAccuracy(self):
        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')
        return cv_results.mean()

    def gridTest(self):
        print("performing grid search...")

        gridSearch = GridSearchCV(estimator=self.classifier,
                                  param_grid=self.gridParams,
                                  cv=self.kfold,
                                  scoring='accuracy',
                                  iid='False',
                                  n_jobs=4)

        gridSearch.fit(self.X, self.y)
        print("best parameters: ", gridSearch.best_params_)
        print("best score: ", gridSearch.best_score_)

    def geneticGridTest(self):
        print("performing Genetic grid search...")

        gridSearch = EvolutionaryAlgorithmSearchCV(estimator=self.classifier,
                                                   params=self.gridParams,
                                                   cv=self.kfold,
                                                   scoring='accuracy',
                                                   verbose=True,
                                                   iid='False',
                                                   n_jobs=4,
                                                   population_size=20,
                                                   gene_mutation_prob=0.30,
                                                   tournament_size=2,
                                                   generations_number=5)
        gridSearch.fit(self.X, self.y)


def main():
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    # create a problem instance:
    test = HyperparameterTuningGrid(RANDOM_SEED)

    print("Default Classifier Hyperparameter values:")
    print(test.classifier.get_params())
    print("score with default values = ", test.getDefaultAccuracy())

    print()
    start = time.time()
    test.gridTest()
    end = time.time()
    print("Time Elapsed = ", end - start)

    print()
    start = time.time()
    test.geneticGridTest()
    end = time.time()
    print("Time Elapsed = ", end - start)


if __name__ == "__main__":
    main()
