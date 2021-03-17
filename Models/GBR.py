#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan
#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Submitted, 2021.

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

class GBR(object):
    def __init__(self):
        self.params_defualt = {'n_estimators': 500,
                                  'max_depth': 5,
                                  'min_samples_split': 5,
                                  'learning_rate': 0.005,
                                  'loss': 'huber'}
        self.tuned_parameters ={'n_estimators': [500],
                                'max_depth': [5],
                                'min_samples_split': [5],
                                'learning_rate': [0.005, 0.01],
                                'loss': ['huber']}

    def auto_tune_params(self, x_train, y_train):
        #use RMSE as the scoring
        clf = GridSearchCV(
            gbr(), self.tuned_parameters, scoring='neg_root_mean_squared_error'
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
        self.model = gbr(**self.params_defualt)

    def model_evaluate(self, x, y, cv):
        scoring = ['neg_root_mean_squared_error', 'r2']
        scores = cross_validate(self.model, x, y, scoring=scoring, cv=cv, return_train_score=True, return_estimator=True)
        self.estimator = scores['estimator']
        return -scores['train_neg_root_mean_squared_error'].mean(), scores['train_r2'].mean(),-scores['test_neg_root_mean_squared_error'].mean(), scores['test_r2'].mean(), scores['estimator']
        #scores1 = cross_val_score(self.model, x, y, cv=cv, scoring='neg_root_mean_squared_error')
        #return scores1,scores1

    def calculate(self, x_train, x_test, y_train, y_test):
        return self.model.feature_importances_


