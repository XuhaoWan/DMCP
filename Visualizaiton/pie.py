#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan
#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Submitted, 2021.

import numpy as np
import matplotlib.pyplot as plt

class plot_pie(object):
    def __init__(self, optimal_model_name, feature_importance):
        font={'family':'Times New Roman', 'weight':'normal', 'size': 18}
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.arange(len(feature_importance)))

        labels = ['e_dp1', 'H_f.ox1', 'N_m1', 'χ_1', 'I_m1', 'r_1', 'N_d1', 'Q_1', 'ΔG_COOH*1', 'ΔG_Max1', 'e_dp2','H_f.ox2', 'N_m2', 'χ_2', 'I_m2', 'r_2', 'N_d2', 'Q_2', 'ΔG_COOH*2', 'ΔG_Max2']
        #for i in range(1,len(feature_importance) + 1):
            #labels.append(str('x')+str(i))


        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)

        wedges, text = ax.pie(feature_importance, colors=colors, shadow=True,
                     startangle=90, textprops=font)
        ax.legend(wedges, labels, bbox_to_anchor=(1, 0, 0, 1), fontsize=8)
        plt.title(optimal_model_name)
        fig.savefig('Pcdac_fipie.jpg')
        plt.show()
