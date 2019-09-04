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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

from pandas import read_csv

class ZooTuning:
    """This class encapsulates the Friedman1 test for a regressor
    """

    VALIDATION_SIZE = 0.90
    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed

        self.initZooDataset()
        #self.initWineDataset()

        self.initClassifier()

        self.X_train, self.X_validation, self.y_train, self.y_validation = \
           model_selection.train_test_split(self.X, self.y, test_size=self.VALIDATION_SIZE, random_state=self.randomSeed)

        #self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)



    def initZooDataset(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'

        self.data = read_csv(url, header=None, usecols=range(1, 18))

        self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.X = self.data.iloc[:, 0:16]
        self.y = self.data.iloc[:, 16]
        #self.X = self.data.iloc[0:60, 0:16]
        #self.y = self.data.iloc[0:60, 16]



    def initWineDataset(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

        self.data = read_csv(url, header=None, usecols=range(0, 14))
        self.X = self.data.iloc[:, 1:14]
        self.y = self.data.iloc[:, 0]

    def initClassifier(self):
        self.classifier = SVC(random_state=self.randomSeed, gamma='scale')

        # self.classifier = DecisionTreeClassifier(random_state=self.randomSeed)
        # self.classifier = AdaBoostClassifier(random_state=self.randomSeed)

        # self.classifier = GradientBoostingClassifier(random_state=self.randomSeed)
        # self.classifier = LogisticRegression(random_state=self.randomSeed, multi_class='auto', solver='lbfgs')
        # self.classifier = KNeighborsClassifier()
        # self.classifier = GaussianNB()

    def __len__(self):
        """
        :return: the total number of items defined in the problem
        """
        return self.X.shape[1]


    def getAccuracyScore(self, params=None):
        if params == None:
            self.initClassifier()
        else:
            C_value = params[0]
            gamma_value = params[1]
            kernel_value = ['linear', 'rbf', 'poly'][round(params[2])]
            degree_value = [round(params[3])]

            #TODO use initClassifier somehow
            self.classifier = SVC(random_state=self.randomSeed,
                                                     C=C_value,
                                                     gamma=gamma_value,
                                                     kernel=kernel_value)


        #cv_results = model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.kfold, scoring='accuracy')
        #return cv_results.mean()

        self.classifier.fit(self.X_train, self.y_train)

        # calculate the regressor's output for the validation set:
        prediction = self.classifier.predict(self.X_validation)

        # return the accuracy score of the predicition vs actual data:
        return accuracy_score(self.y_validation, prediction)



# testing the class:
def main():
    # create a problem instance:
    zoo = ZooTuning(42)

    print("Classifier: ", zoo.classifier.get_params())
    print("score (default) = ", zoo.getAccuracyScore())

    # Create range of candidate penalty hyperparameter values
    #penalty = ['l1', 'l2']  # Create range of candidate regularization hyperparameter values C
    # Choose 10 values, between 0 and 4
    #C = np.logspace(0, 4, 10)  # Create dictionary hyperparameter candidates
    #hyperparameters = dict(C=C, penalty=penalty)  # Create grid search, and pass in all defined values

    print("performing grid search...")
    grid_param = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],            #np.logspace(0, 10, 30),
        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5], #np.logspace(0, 2, 30),
        'degree': [0, 1, 2, 3, 4, 5, 6]
    }

    gd_sr = GridSearchCV(estimator=zoo.classifier,
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=zoo.kfold,
                         iid='False',
                         n_jobs=-1)

    gd_sr.fit(zoo.X, zoo.y)
    print("best parameters: ", gd_sr.best_params_)
    print("best score: ", gd_sr.best_score_)

    #TODO:
    # print("Confusion Matrix:\n", metrics.confusion_matrix(prediction, test_y))

    #print("score (1.0, 'scale') = ", zoo.getAccuracyScore([1.0, 'scale']))
    #print("score (1.0, 'auto') = ", zoo.getAccuracyScore([1.0, 'auto']))
    #print("optimized score = ", zoo.getAccuracyScore([1.9, 0.236]))


if __name__ == "__main__":
    main()
