from sklearn import model_selection
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings

from math import floor

class MlpHyperparametersTest:

    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initDataset()
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

    def initDataset(self):
        self.data = datasets.load_iris()

        self.X = self.data['data']
        self.y = self.data['target']


    # params contains floats representing the following:
    # 'hidden_layer_sizes': up to 4 positive integers
    # 'activation': {'tanh', 'relu', 'logistic'},
    # 'solver': {'sgd', 'adam', 'lbfgs'},
    # 'alpha': float,
    # 'learning_rate': {'constant', 'invscaling', 'adaptive'}
    def convertParams(self, params):

        # transform the layer sizes from float (possibly negative) values into hiddenLayerSizes tuple:
        if round(params[1]) <= 0:
            hiddenLayerSizes = round(params[0]),
        elif round(params[2]) <= 0:
            hiddenLayerSizes = (round(params[0]), round(params[1]))
        elif round(params[3]) <= 0:
            hiddenLayerSizes = (round(params[0]), round(params[1]), round(params[2]))
        else:
            hiddenLayerSizes = (round(params[0]), round(params[1]), round(params[2]), round(params[3]))

        activation = ['tanh', 'relu', 'logistic'][floor(params[4])]
        solver = ['sgd', 'adam', 'lbfgs'][floor(params[5])]
        alpha = params[6]
        learning_rate = ['constant', 'invscaling', 'adaptive'][floor(params[7])]

        return hiddenLayerSizes, activation, solver, alpha, learning_rate

    @ignore_warnings(category=ConvergenceWarning)
    def getAccuracy(self, params):
        hiddenLayerSizes, activation, solver, alpha, learning_rate = self.convertParams(params)

        self.classifier = MLPClassifier(random_state=self.randomSeed,
                                        hidden_layer_sizes=hiddenLayerSizes,
                                        activation=activation,
                                        solver=solver,
                                        alpha=alpha,
                                        learning_rate=learning_rate)

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')

        return cv_results.mean()

    def formatParams(self, params):
        hiddenLayerSizes, activation, solver, alpha, learning_rate = self.convertParams(params)
        return "'hidden_layer_sizes'={}\n " \
               "'activation'='{}'\n " \
               "'solver'='{}'\n " \
               "'alpha'={}\n " \
               "'learning_rate'='{}'"\
            .format(hiddenLayerSizes, activation, solver, alpha, learning_rate)
