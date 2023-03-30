import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(root_path,data_path):
    dataset = pd.read_excel(os.path.join(root_path,data_path),index_col=0)
    print("load done! dataset shape {}".format(dataset.shape))
    return dataset

def train_valid_test_split(dataset,split_rate,selected_features):
    dataset = dataset[selected_features]
    days = len(dataset)/24
    print("days:{}".format(days))
    train_days = int(days*split_rate[0])
    test_days  = int(days*(1-split_rate[1]))
    valid_days = int(days-train_days-test_days)
    print("train dataset includes {} days".format(train_days))
    print("valid dataset includes {} days".format(valid_days))
    print("test  dataset includes {} days".format(test_days))
    train = dataset[:train_days * 24]
    valid = dataset[train_days * 24:train_days * 24 + valid_days *24]
    test  = dataset[-(test_days * 24):]
    print("train:{} valid:{} test:{}".format(len(train),len(valid),len(test)))
    min_max_scaler = MinMaxScaler()
    train = min_max_scaler.fit_transform(train.values)
    valid = min_max_scaler.transform(valid.values)
    test = min_max_scaler.transform(test.values)
    return train,valid,test

class mydataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        inputs = self.X[item]
        target = self.y[item]
        return inputs,target

def create_dataset_multi_steps(dataset,seq_len,pred_len,slide_width,is_train):
    if is_train == True:
        X,y = [],[]
        in_start = 0
        for _ in range(len(dataset)):
            in_end = in_start + seq_len
            out_end = in_end + pred_len
            if out_end < len(dataset):
                x_input = dataset[in_start:in_end,:]
                #x_input = x_input.reshape(len(x_input),1)
                X.append(x_input)
                y.append(dataset[in_end:out_end,-1])
            in_start += slide_width
        X = np.array(X)
        y = np.array(y)
        y = y.reshape(y.shape[0],y.shape[1])
        return X,y
    else:
        X,y = [],[]
        in_start = 0
        for _ in range(len(dataset)):
            in_end = in_start + seq_len
            out_end = in_end + pred_len
            if out_end < len(dataset):
                x_input = dataset[in_start:in_end,:]
                #x_input = x_input.reshape(len(x_input),1)
                X.append(x_input)
                y.append(dataset[in_end:out_end,-1])
            in_start += 1
        X = np.array(X)
        y = np.array(y)
        y = y.reshape(y.shape[0],y.shape[1])
        return X,y




def data_provider(args,
                  flag):
    dataset = load_data(args.root_path,args.data_path)
    train,valid,test = train_valid_test_split(dataset,args.split_rate,args.selected_features)
    train_X, train_y = create_dataset_multi_steps(train, args.seq_len, args.pred_len, args.slide_width, is_train=True)
    valid_X, valid_y = create_dataset_multi_steps(train, args.seq_len, args.pred_len, args.slide_width, is_train=False)
    test_X, test_y = create_dataset_multi_steps(train, args.seq_len, args.pred_len, args.slide_width, is_train=False)
    train_data = mydataset(train_X, train_y)
    valid_data = mydataset(valid_X, valid_y)
    test_data = mydataset(test_X, test_y)
    train_dataiter = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataiter = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)
    test_dataiter = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    if flag == "train":
        return train_data,train_dataiter
    elif flag == "val":
        return valid_data,valid_dataiter
    else:
        return test_data,test_dataiter

if __name__ == "__main__":
    root_path = '/Users/gaohaitao/Desktop/毕业论文/code/pv_model/pv_forecasting_model/data/'
    data_path = 'solarpower.xlsx'
    split_rate = [0.6,0.8]
    selected_features = ['power','max_temp','next_max_temp']
    seq_len = 24
    pred_len = 24
    slide_width = 1
    batch_size = 32
    train_dataset,train_dataiter = data_provider(root_path,
                                                 data_path,
                                                 split_rate,
                                                 selected_features,
                                                 seq_len,
                                                 pred_len,
                                                 slide_width,
                                                 batch_size,
                                                 flag='train')


