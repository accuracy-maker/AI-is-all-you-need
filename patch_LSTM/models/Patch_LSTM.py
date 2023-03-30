import torch
import torch.nn as nn
import copy
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        # RevIn (Reversible Instance Normalization ) 数据处理的手法，规范化
        self.revin = configs.revin
        self.c_in = configs.enc_in
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last
        if self.revin: self.revin_layer = RevIN(self.c_in, affine=self.affine,
                                                subtract_last=self.subtract_last)
        #patch
        self.patch = 24
        #extract features
        self.lstm = nn.LSTM(input_size=self.patch,hidden_size=configs.hidden_size,
                            num_layers=2,
                            dropout=configs.dropout)
        #output
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(in_features=int((configs.seq_len / self.patch)*configs.hidden_size),
                                out_features=configs.pred_len)

    def forward(self,x):
        # norm x: [Batch_size,seq_len,nvars]
        x = x.permute(0,2,1)

        if self.revin:
            x = x.permute(0,2,1) #x: [Batch_size,nvars,seq_len]
            x = self.revin_layer(x, 'norm')
            # x = x.permute(0,2,1) #x: [Batch_size,seq_len,nvars]
        #do patching
        b,l,n = x.size()
        x = x.permute(0,2,1)#x: [Batch_size,nvars,seq_len]
        x = x.reshape(b*n,-1,self.patch)
        #extract features: [Linear,CNN,LSTM]
        x,_ = self.lstm(x) # x:[Batch_size*nvars, 4,512]

        #model: [CNN,LSTM]
        x = self.flatten(x) # x:[Batch_size*nvars, 4*512]
        x = x.reshape(b,n,-1)# x:[Batch_size,nvars, 4*512]
        x = self.linear(x)

        #denorm
        if self.revin:
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0,2,1)

        return x.permute(0,2,1)

