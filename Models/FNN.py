#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: xhwan
#If you use DMCP in your research, please cite the following paper:X. Wan, Z. Zhang*, W. Yu, Y. Guo*, A State-of-the-art Density-functional-theory-based and Machine-learning-accelerated Hybrid Method for Intricate System Catalysis. Submitted, 2021.

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torchvision import datasets, transforms
from torch.nn import init
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


class FNN(object):
    def __init__(self, x_train, x_test, y_train, y_test, params, model_evaluation, x, y):
        self.x = x
        self.y = y
        self.x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
        self.x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
        y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
        self.y_test = torch.unsqueeze(y_test, 1)
        self.y_train = torch.unsqueeze(y_train, 1)
        self.params_defualt = {'BATCH_SIZE': 32,
                                'LR': 0.05,
                                'EPOCH': 50}
        self.params_modify = params
        self.modify_params()
        torch_data = Data.TensorDataset(self.x_train, self.y_train)
        self.loader = Data.DataLoader(dataset=torch_data, batch_size=self.params_defualt['BATCH_SIZE'], shuffle=True)
        self.calculate()

    def modify_params(self):
        for key in self.params_modify:
            self.params_defualt[key] = self.params_modify[key]

    def calculate(self):
        adam_net = Net()

        opt_adam = torch.optim.Adam(adam_net.parameters(), lr=self.params_defualt['LR'])
        loss_func = nn.MSELoss()

        all_loss = {}
        for epoch in range(self.params_defualt['EPOCH']):
            print('epoch', epoch)
            for step, (b_x, b_y) in enumerate(self.loader):
                print('step', step)
                pre = adam_net(b_x)
                loss = loss_func(pre, b_y)
                opt_adam.zero_grad()
                loss.backward()
                opt_adam.step()
                all_loss[epoch + 1] = loss
        print(all_loss)

        yt = self.y_train.numpy()
        yp = adam_net(self.x_train)
        yp = yp.detach().numpy()
        rmse = np.sqrt(mse(yt, yp))
        r2 = r2_score(yt, yp)
        yt1 = self.y_test.numpy()
        yp1 = adam_net(self.x_test)
        yp1 = yp1.detach().numpy()
        rmset = np.sqrt(mse(yt1, yp1))
        r2t = r2_score(yt1, yp1)
        print(rmse)
        print(r2)
        print(rmset)
        print(r2t)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(20, 32)
        self.predict = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight.data)



