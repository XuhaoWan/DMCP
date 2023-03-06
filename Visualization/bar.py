#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xuhao wan, wei yu
#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Materials Reports: Energy. 2021.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

class plot_bar(object):
    def __init__(self, title, train_set_RMSE, train_set_R2, test_set_RMSE, test_set_R2):
        data_list2 = []
        for item in [train_set_RMSE, train_set_R2, test_set_RMSE, test_set_R2]:
            data_list1 = []
            model_list = []
            for i in item.keys():
                data_list1.append(mean(item[i]))
                model_list.append(i)
            data_list2.append(data_list1)
        font={'family':'Times New Roman', 'weight':'normal', 'size':20}
        ##取出 4个dict中的数据，注意dict中的数据存放是无序的
        R2train = data_list2[1]  # train set R2 score
        R2test = data_list2[3]# test set R2 score
        RMSEtrain = data_list2[0]  # train set RMSE
        RMSEtest = data_list2[2] # test set RMSE
        #for item in [R2train, R2test, RMSEtrain, RMSEtest]:
            #for i in range(len(item)):
                #if item[i] < 0:
                    #item[i] = 0.5
        label = model_list
        bar_width = 0.4
        bar_x = np.arange(len(label))


        fig1 = plt.figure(figsize=(9, 6))
        ax1 = fig1.add_subplot(111)
        #ax1.set_title('RMSE')
        bar1 = ax1.bar(x=bar_x - bar_width/2,   # 设置不同的x起始位置
                      height= RMSEtrain, width=bar_width, color='royalblue')
        bar2 = ax1.bar(x=bar_x + bar_width/2,   # 设置不同的x起始位置
                      height= RMSEtest, width=bar_width, color='darkorange'
                )

        ax1.set_ylabel('RMSE /eV', fontsize=24, fontfamily='Times New Roman')
        ax1.set_xticks(range(len(label)))
        ax1.set_xticklabels(label, fontsize=20, fontfamily='Times New Roman')
        #ax1.set_yticklabels(np.around((np.arange(0, 0.4, 0.05)), decimals=2), fontsize=20, fontfamily='Times New Roman')
        ax1.legend((bar1, bar2), ('Train set', 'Test set'), prop=font)

        fig2 = plt.figure(figsize=(9, 6))
        ax2 = fig2.add_subplot(111)
        #ax1.set_title('RMSE')
        bar1 = ax2.bar(x=bar_x - bar_width/2,   # 设置不同的x起始位置
                      height= R2train, width=bar_width, color='royalblue')
        bar2 = ax2.bar(x=bar_x + bar_width/2,   # 设置不同的x起始位置
                      height= R2test, width=bar_width, color='darkorange'
                )

        ax2.set_ylabel('Score', fontsize=24, fontfamily='Times New Roman')
        ax2.set_xticks(range(len(label)))
        ax2.set_xticklabels(label, fontsize=20, fontfamily='Times New Roman')
        #ax2.set_yticklabels(np.around((np.arange(0, 1.0, 0.2)), decimals=2), fontsize=20, fontfamily='Times New Roman')
        ax2.legend((bar1, bar2), ('Train set', 'Test set'), prop=font)

        fig1.savefig('bar_RMSE.jpg')
        fig2.savefig('bar2_R2.jpg')
        plt.show()
