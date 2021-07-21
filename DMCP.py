#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xuhao wan, wei yu
#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Materials Reports: Energy. 2021.

from Models.RFR import RFR
from Models.KRR import KRR
from Models.GBR import GBR
from Models.KNR import KNR
from Models.FNN import FNN
from Models.SVR import SVR
from Models.Lasso import LSO
from Models.ENR import ENR
from Models.GPR import GPR
from Models.ETR import ETR
from Models.MLP import MLP
from Visualization.Violin import plot_Violin
from Visualization.bar import plot_bar
from Visualization.scatter import plot_scatter
from Visualization.pearson import plot_pearson
from Visualization.pie import plot_pie
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import f90nml
from multidict import CIMultiDict
from statistics import mean

input_file = f90nml.read('DMCP_input_file')
data = CIMultiDict(input_file['data'])
general = CIMultiDict(input_file['general'])
visualization = CIMultiDict(input_file['visualization'])


def parse_data():
    # intrn
    if 'intrn' not in data.keys():
        print('No train data file')
    else:
        data_file = data['intrn']
        data_train = np.loadtxt(data_file, delimiter=",", dtype="float")
        # grept
        if 'grept' not in general.keys():
            iteration = 1
        else:
            iteration = general['grept']

        train_set_RMSE = {}
        train_set_R2 = {}
        test_set_RMSE = {}
        test_set_R2 = {}
        estimator_dict = {}
        for i in range(iteration):
            x = preprocessing_data(data_train)
            x = add_noise(x)
            y = data_train[..., -1]
            ##psplt
            if 'psplt' not in general.keys():
                test_size = 0.2
            else:
                test_size = 1 - general['psplt']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=16)

            # model_load
            # gmodl
            if 'gmodl' not in general:
                print("Not choose model")
            else:
                model_list = general['gmodl']
                if type(model_list) is str:
                    model_list = [model_list]
                for model in model_list:
                    param = get_params(model)
                    model_obj = eval(model)
                    ML_model = model_obj()
                    if 'modpr' in general:
                        if general['modpr'] == 'ON':
                            ML_model.auto_tune_params(x_train, y_train)
                    ML_model.modify_params(param)
                    ML_model.build_model()
                    # gcrva
                    if 'gcrva' in general.keys():
                        if general['gcrva'] == 'ON':
                            train_rmse, train_r2, test_rmse, test_r2, estimator = ML_model.model_evaluate(x, y, general[
                                'gcvrn'])
                            if i == 0:
                                train_set_RMSE[model] = [train_rmse]
                                train_set_R2[model] = [train_r2]
                                test_set_RMSE[model] = [test_rmse]
                                test_set_R2[model] = [test_r2]
                                estimator_dict[model] = [estimator]
                            else:
                                train_set_RMSE[model].append(train_rmse)
                                train_set_R2[model].append(train_r2)
                                test_set_RMSE[model].append(test_rmse)
                                test_set_R2[model].append(test_r2)
                                estimator_dict[model].append(estimator)
                    # ML_model.calculate(x_train, x_test, y_train, y_test)
        result_visualize(train_set_RMSE, train_set_R2, test_set_RMSE, test_set_R2)
        optimal_model, optimal_model_name = choose_optimal_model(train_set_RMSE, estimator_dict)
        predict(optimal_model, optimal_model_name)


def predict(optimal_model, optimal_model_name):
    if 'intrn' not in data.keys():
        print('No train data file')
    else:
        data_file = data['intrn']
        data_train = np.loadtxt(data_file, delimiter=",", dtype="float")
        x = preprocessing_data(data_train)
        x = add_noise(x)
        y = data_train[..., -1]
        ##psplt
        if 'psplt' not in general.keys():
            test_size = 0.2
        else:
            test_size = 1 - general['psplt']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=16)
        y_train_true, y_train_pred = y_train, optimal_model.predict(x_train)
        y_test_true, y_test_pred = y_test, optimal_model.predict(x_test)
        plot_scatter(y_train_true, y_train_pred, y_test_true, y_test_pred)
        plot_pearson(optimal_model_name, x_train)
        if optimal_model_name in ['GBR', 'RFR', 'ETR']:
            plot_pie(optimal_model_name, optimal_model.feature_importances_)


def choose_optimal_model(model_evaluate_data, estimator_dict):
    if 'fmodl' in visualization:
        optimal_model_name = visualization['fmodl']
    else:
        model_list = []
        data_list = []
        for key in model_evaluate_data.keys():
            model_list.append(key)
            data_list.append(mean(model_evaluate_data[key]))
        optimal_model_name = model_list[data_list.index(min(data_list))]
    optmal_index = model_evaluate_data[optimal_model_name].index(min(model_evaluate_data[optimal_model_name]))
    optimal_model = estimator_dict[optimal_model_name][optmal_index][0]
    return optimal_model, optimal_model_name


# def parse_DMCP_input_file():
# input_file = f90nml.read('DMCP_input_file')
# return input_file

def result_visualize(train_set_RMSE, train_set_R2, test_set_RMSE, test_set_R2):
    if 'vvoln' in visualization.keys():
        if visualization['vvoln'] == 'ON':
            plot_Violin('RMSE', test_set_RMSE)
            plot_Violin('R2', test_set_R2)
    if 'vcomp' in visualization.keys():
        if visualization['vcomp'] == 'ON':
            plot_bar('DMCP', train_set_RMSE, train_set_R2, test_set_RMSE, test_set_R2)


def preprocessing_data(data_train):
    # pscal
    if 'pscal' not in general.keys():
        X = data_train[..., 0:(data_train.shape[1] - 1)]
    else:
        if general['pscal'] == 'OFF':
            X = data_train[..., 0:(data_train.shape[1] - 1)]
        elif general['pscal'] == 'NOR':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(data_train[..., 0:(data_train.shape[1] - 1)])
        elif general['pscal'] == 'STA':
            scaler = StandardScaler()
            X = scaler.fit_transform(data_train[..., 0:(data_train.shape[1] - 1)])
        else:
            scaler = Normalizer(norm='l2')
            X = scaler.fit_transform(data_train[..., 0:(data_train.shape[1] - 1)])
    return X


def add_noise(X):
    # pnose
    if 'pnose' not in general.keys():
        X = X
    else:
        scale = general['pnose']
        x_noise = np.random.normal(loc=0.0, scale=scale, size=X.shape)
        X = X + x_noise
    return X


def get_params(model):
    if ('PR' + model) not in general.keys():
        param = {}
        print("Not set" + 'PR' + model)
    else:
        param = general['PR' + model]
        param_key = param[1:(len(param) - 1):3]
        param_val = param[3:(len(param) - 1):3]
        param = dict(zip(param_key, param_val))
    return param


def main():
    parse_data()

if __name__ == "__main__":
    main()
