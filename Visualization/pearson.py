#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xuhao wan, wei yu
#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Materials Reports: Energy. 2021.

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

class plot_pearson(object):
    def __init__(self, optimal_model_name,corelation):
        pearson_r2 = []
        for i in range(corelation.shape[1]):
            pearson_r1 = []
            for j in range(corelation.shape[1]):
                r, _ = pearsonr(corelation[:][i], corelation[:][j])
                pearson_r1.append(r)
            pearson_r2.append(pearson_r1)

        ax1 = sns.heatmap(pearson_r2, vmin=-1, vmax=1, cmap='RdBu')
        plt.title(optimal_model_name)
        plt.savefig('Pcdac_pearson.jpg')
        plt.show()


