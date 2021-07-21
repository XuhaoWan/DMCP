#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xuhao wan, wei yu
#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Materials Reports: Energy. 2021.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class plot_scatter(object):
    def __init__(self, y_train_true, y_train_pred, y_test_true, y_test_pred):
        font={'family':'Times New Roman', 'weight':'normal', 'size': 24}
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)

        dot1 = ax.scatter(y_train_true, y_train_pred,
                   s=80, c='white', edgecolors='royalblue', marker='o', linewidth=2)
        dot2 = ax.scatter(y_test_true, y_test_pred,
                   s=80, c='white', edgecolors='darkorange', marker='s', linewidth=2)
        line = ax.plot([0,1,2.2], [0,1,2.2], color='k')

        ax.set_xlabel('$\mathregular{G_{DFT}}$ /eV', fontsize=24, fontfamily='Times New Roman')
        ax.set_ylabel('$\mathregular{G_{ML}}$ /eV' , fontsize=24, fontfamily='Times New Roman')
        ax.set_xlim(xmin=0, xmax=2.2)
        ax.set_ylim(ymin=0, ymax=2.2)
        ax.set_xticklabels(np.around((np.arange(0, 2.2, 0.25)), decimals=2), fontsize=20, fontfamily='Times New Roman')
        ax.set_yticklabels(np.around((np.arange(0, 2.2, 0.25)), decimals=2), fontsize=20, fontfamily='Times New Roman')
        ax.legend((dot1, dot2), ('Train set', 'Test set'), prop=font)

        fig.savefig('Pcdac_scatter.jpg', bbox_inches='tight')
        plt.show()
