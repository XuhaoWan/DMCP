#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Submitted, 2021.

import numpy as np
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_validate,GridSearchCV
from sklearn.preprocessing import MinMaxScaler


class MLP(object):
    def __init__(self):
        self.params_defualt = {'learning_rate': 'constant',
                                 'learning_rate_init': 0.001,
                                 'batch_size': 4,
                                 'hidden_layer_sizes': (20, 32),
                                 'random_state': 1,
                                 'max_iter': 100000,
                                 'activation': 'logistic'}
        self.tuned_parameters = {'learning_rate':['constant'],
                               'learning_rate_init':[0.001],
                               'batch_size':[4],
                                'hidden_layer_sizes':[(20, 32)],
                                'random_state':[1],
                                'max_iter':[100000],
                                'activation':['logistic']}

    def auto_tune_params(self, x_train, y_train):
        #use RMSE as the scoring
        clf = GridSearchCV(
            mlp(), self.tuned_parameters, scoring='neg_root_mean_squared_error'
        )
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        self.params_defualt = clf.best_params_

    def modify_params(self, params):
        for key in params:
            self.params_defualt[key] = params[key]

    def build_model(self):
        self.model = mlp(**self.params_defualt)

    def model_evaluate(self, x, y, cv):
        scoring = ['neg_root_mean_squared_error', 'r2']
        scores = cross_validate(self.model, x, y, scoring=scoring, cv=cv, return_train_score=True,
                                return_estimator=True)
        self.estimator = scores['estimator']
        return -scores['train_neg_root_mean_squared_error'].mean(), scores['train_r2'].mean(), -scores[
            'test_neg_root_mean_squared_error'].mean(), scores['test_r2'].mean(), scores['estimator']

    def calculate(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)
        rmse = np.sqrt(mse(y_train, self.model.predict(x_train)))
        r2 = r2_score(y_train, self.model.predict(x_train))
        rmset = np.sqrt(mse(y_test, self.model.predict(x_test)))
        r2t = r2_score(y_test, self.model.predict(x_test))
        print('pre:', self.model.predict(x_test))
        print(y_test)
        print(rmse)
        print(r2)
        print(rmset)
        print(r2t)
        return r2
