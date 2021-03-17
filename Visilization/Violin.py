#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan
#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Submitted, 2021.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class plot_Violin(object):
    def __init__(self, title, data_dict):
        model_list = []
        data_list = []
        #for key in data_dict.keys():
            #model_list.append(key)
            #data_list.append(data_dict[key])
        tips = pd.DataFrame.from_dict(data_dict)
        #tips = np.array(data_list).T
        sns.set(style='ticks', font='Times New Roman', font_scale=1.8)
        fig = plt.figure(figsize=(9, 6))
        ax=sns.violinplot(data=tips,
                       split=True,
                       linewidth = 2, #线宽
                       width = 0.8,   #箱之间的间隔比例
                       palette = 'pastel', #设置调色板
                       #order = model_list, #筛选类别
                       # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                       gridsize = 50, #设置小提琴图的平滑度，越高越平滑
                       # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                       #bw = 0.8      #控制拟合程度，一般可以不设置
                       )
        ax.set_ylabel(title + 'Score', fontsize=28, fontfamily='Times New Roman')
        ax.set_xlabel('model', fontsize=28, fontfamily='Times New Roman')
        #ax = fig.add_subplot(111)

        fig.savefig('volin_' + title + '.jpg', bbox_inches='tight')
        plt.show()

#data_dict = {'GBR': [-0.32744250093837, -0.3958306749566164], 'KNR': [-0.3522280594983168, -0.3593963196775159], 'SVR': [-0.34374401871141413, -0.34763844616449313], 'GPR': [-0.3802567562102421, -0.3864921408452695], 'MLP': [-0.3912175478421521, -0.39099797659443214], 'RFR': [-0.34453840142807285, -0.3616486858237452], 'ETR': [-0.41477900933292655, -0.49834959923665945], 'KRR': [-0.4050557476038392, -0.41178328305680595], 'LSO': [-0.3890709246795953, -0.3890709246795953], 'ENR': [-0.3890709246795953, -0.3890709246795953]}
#data_show = plot_Violin(data_dict)
