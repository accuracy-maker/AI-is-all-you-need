import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import *
from sklearn.metrics import mean_squared_error
import os
import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def my_mape(test_y_sum,y_pred_sum):

    a = np.abs(test_y_sum-y_pred_sum)
    b=a/test_y_sum
    c=np.sum(b,axis=0)/len(b)*100
    return c

def hour_mae(y_true,y_pred):
    minus = np.abs(y_pred-y_true)
    mae = np.mean(minus,axis=0)
    return mae





name_list = ['三甲晨光对面','台州椒江下陈明星光伏基站','温岭新河铁场村-2','温岭泽国高坦村','路桥新桥下林桥','路桥新桥桥头叶','路桥枧头林村',
                 '路桥金清富强村-2','路桥金清霓岙','路桥蓬街光明村-2']
for i in range(len(name_list)):
    pth_path = './training/models/model'+str(i)+'.pth'
    model = torch.load(pth_path)
    y_pred = model.test()
    for _ in range(len(y_pred)):
        if y_pred[_] < 0:
            y_pred[_] = 0
    y_pred = y_pred.reshape(-1, 24)
    y_true = model.y[model.train_timesteps:].reshape(-1, 24)
    y_pred_sum = np.sum(y_pred, axis=1)
    y_true_sum = np.sum(y_true, axis=1)
    mape = my_mape(y_pred_sum, y_true_sum)
    print("mape={}".format(mape))
    mae = hour_mae(y_true, y_pred)
    print("mae={}".format(mae))
    mse = mean_squared_error(y_pred.reshape(-1), y_true.reshape(-1))
    print("mse={}".format(mse))
    #对真实值进行反归一化
    scaler_path = './training/scaler_args_save/target_scaler'+str(i)+'.save'
    scaler = joblib.load(scaler_path)
    y_pred = y_pred.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_true_inverse = scaler.inverse_transform(y_true)
    print("反归一化完成...")
    y_pred_inverse = y_pred_inverse.reshape(-1)
    y_true_inverse = y_true_inverse.reshape(-1)

    fig1 = plt.figure()# 真实值与预测值的对比
    plt.plot(y_true_inverse, 'b', label="True")
    plt.plot(y_pred_inverse, 'g', label='Predicted')
    plt.legend(loc='upper left')
    plt.title("Prediction of Active Power")
    plt.xlabel("hours")
    plt.ylabel("Active Power Consumption")
    figpath1 = './prediction/prediction_figures/'+name_list[i]+'predition_figure.png'
    plt.savefig(figpath1, dpi=120)
    plt.close(fig1)

    fig2 = plt.figure()#每个小时的相对误差
    plt.bar(range(len(mae)), mae)
    plt.title("24 hours respective MAE")
    plt.xlabel("hours")
    plt.ylabel("MAE")
    figpath2 = './prediction/prediction_figures/' + name_list[i] + 'mae_figure.png'
    plt.savefig(figpath2, dpi=120)
    plt.close(fig2)
