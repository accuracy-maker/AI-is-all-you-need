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
def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="DA-RNN")
    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="./dataset/df.csv", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=512, help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=256, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=256, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=24, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')

    # parse the arguments
    args = parser.parse_args()

    return args

def main():
    """Main pipeline of DA-RNN."""
    args = parse_args()
    path = './training/'
    # Read dataset
    print("==> Load dataset ...")
    # X, y = read_data(args.dataroot, debug=False)
    df = pd.read_csv(args.dataroot)
    name_list = ['三甲晨光对面','台州椒江下陈明星光伏基站','温岭新河铁场村-2','温岭泽国高坦村','路桥新桥下林桥','路桥新桥桥头叶','路桥枧头林村',
                 '路桥金清富强村-2','路桥金清霓岙','路桥蓬街光明村-2']
    datas = list(df.groupby(df.group))
    for i in range(len(name_list)):
        print("==>正在读取{}基站的数据...".format(name_list[i]))
        data = datas[i][1].iloc[:,[0,5]].copy()#选取历史值和天气作为feature
        X = np.array(data)
        y = np.array(data.iloc[:,0]).reshape(-1,1)
        print("==>对{}基站数据进行归一化...")
        print("==>生成训练数据归一化器scaler1...")
        scaler1 = MinMaxScaler((0,1))
        print("生成完毕...")
        print("==>生成target数据归一化器scaler2...")
        scaler2 = MinMaxScaler((0,1))
        print("生成完毕...")
        train_X = X[:int((X.shape[0]*0.7//24)*24)]
        test_X = X[int((X.shape[0]*0.7//24)*24):]
        train_y = y[:int((X.shape[0]*0.7//24)*24)]
        test_y = y[int((X.shape[0]*0.7//24)*24):]
        train_X = scaler1.fit_transform(train_X)
        test_X = scaler1.transform(test_X)
        train_y = scaler2.fit_transform(train_y)
        test_y = scaler2.transform(test_y)
        X = np.concatenate((train_X,test_X),axis=0)
        y = np.concatenate((train_y,test_y))
        y = y.reshape(-1)

        print('正在准备保存归一化器参数...')
        # 保存归一化模型
        target_scaler_filname = 'target_scaler'+str(i)+'.save'
        Test_PATH = './training/' + target_scaler_filname
        # 保存训练归一化器
        joblib.dump(scaler2, Test_PATH)
        print("预测数据归一化保存完毕！")


        # Initialize model
        print("==> Initialize DA-RNN model ...")
        model = DA_RNN(
            X,
            y,
            args.ntimestep,
            args.nhidden_encoder,
            args.nhidden_decoder,
            args.batchsize,
            args.lr,
            args.epochs
        )

        # Train
        print("==> Start training ...")
        model.train()



        fig1 = plt.figure()
        plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
        plt.title("iter_losses")
        plt.xlabel("number of iters")
        plt.ylabel("Loss")
        figname1 = path+'figures/'+name_list[i]+'iter_losses.png'
        plt.savefig(figname1,dpi=120)
        plt.close(fig1)

        fig2 = plt.figure()
        plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
        plt.title("Epoch Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        figname2 = path+'figures/'+name_list[i]+'epochs_losses.png'
        plt.savefig(figname2,dpi=120)
        plt.close(fig2)


        print("保存基站的模型参数....")
        model_name = path+'models/model'+str(i)+'.pth'
        torch.save(obj=model,f=model_name)
        print("保存基站的模型参数成功！")
        print("基站执行完毕".format(name_list[i]))
if __name__ == '__main__':
    main()
