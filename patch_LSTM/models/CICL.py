import torch
import torch.nn as nn
import copy

def clones(module,N):
    """用于生成相同网络层的克隆函数，它的参数module表示要克隆的目标网络层，N代表需要克隆的数量"""
    #在函数中，我们通过for循环对module进行N次深度拷贝，使其每个module成为独立的层
    #然后将其放到nn.ModuleList类型的列表中存放
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def compute_shape(x,conv_kernel,conv_stride,mp_kernel,mp_stride):
    x = (x-(conv_kernel-1)-1)/conv_stride+1
    x = (x-(mp_kernel-1)-1)/mp_stride+1
    return int(x)


class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.input_size = len(configs.selected_features)
        self.hidden_size1 = configs.hidden_size[0]
        self.hidden_size2 = configs.hidden_size[1]
        self.LSTM1 = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size1,dropout=configs.dropout)
        self.LSTM2 = nn.LSTM(input_size=self.hidden_size1, hidden_size=self.hidden_size2, dropout=configs.dropout)
        self.conv_kernel = configs.conv_kernel
        self.conv_stride = configs.conv_stride
        self.mp_kernel = configs.max_pooling_kernel
        self.mp_stride = configs.max_pooling_stride
        self.l_out1 = compute_shape(configs.seq_len,self.conv_kernel,self.conv_stride,self.mp_kernel,self.mp_stride)
        self.l_out2 = compute_shape(self.l_out1,self.conv_kernel,self.conv_stride,self.mp_kernel,self.mp_stride)

        self.conv1 = nn.Conv1d(in_channels=self.hidden_size2,out_channels=64,kernel_size=self.conv_kernel,stride=self.conv_stride)
        self.max_pooling = nn.MaxPool1d(kernel_size=self.mp_kernel,stride=self.mp_stride)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=32,kernel_size=self.conv_kernel,stride=self.conv_stride)
        self.max_pooling = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=self.mp_stride)
        self.linear1 = nn.Linear(in_features=32,out_features=1024,bias=True)
        self.linear2 = nn.Linear(in_features=1024*self.l_out2,out_features=configs.pred_len,bias=True)
        self.dropout = nn.Dropout(p=configs.dropout)
        self.relu = nn.ReLU()

    def forward(self,x):
        x,_ = self.LSTM1(x.permute(1,0,2))
        x,_ = self.LSTM2(self.relu(x))
        x = self.conv1(x.permute(1,2,0))
        x = self.max_pooling(x)
        x = self.conv2(x)
        x = self.dropout(self.max_pooling(x))
        x = self.linear1(x.transpose(1,2))
        N,H,L = x.size()
        x = x.reshape(N,-1)
        x = self.linear2(self.relu(x))
        x = x.unsqueeze(-1)
        return x



