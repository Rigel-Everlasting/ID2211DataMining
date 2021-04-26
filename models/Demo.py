#!/usr/bin/env python  
#-*- coding:utf-8 -*-  
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: demo.py 
# @time: 2021/04/25
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
# from data_loader import DigitalMusicDataset
from Data import *
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from torch.nn import MSELoss

rt = load_data()
userNum = rt['reviewerID'].max() + 1
itemNum = rt['asin'].max() + 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SVD(Module):

    def __init__(self,userNum,itemNum,dim):
        super(SVD, self).__init__()
        self.uEmbd = nn.Embedding(userNum,dim)
        self.iEmbd = nn.Embedding(itemNum,dim)
        self.uBias = nn.Embedding(userNum,1)
        self.iBias = nn.Embedding(itemNum,1)
        self.overAllBias = nn.Parameter(torch.Tensor([0]))

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        ubias = self.uBias(userIdx)
        ibias = self.iBias(itemIdx)

        biases = ubias + ibias + self.overAllBias
        prediction = torch.sum(torch.mul(uembd,iembd),dim=1) + biases.flatten()

        return prediction


class NCF(Module):

    def __init__(self,userNum,itemNum,dim,layers=[128,64,32,8]):
        super(NCF, self).__init__()
        self.uEmbd = nn.Embedding(userNum,dim)
        self.iEmbd = nn.Embedding(itemNum,dim)
        self.fc_layers = torch.nn.ModuleList()
        self.finalLayer = torch.nn.Linear(layers[-1],1)

        for From,To in zip(layers[:-1],layers[1:]):
            self.fc_layers.append(nn.Linear(From,To))

    def forward(self, userIdx,itemIdx):
        uembd = self.uEmbd(userIdx)
        iembd = self.iEmbd(itemIdx)
        embd = torch.cat([uembd, iembd], dim=1)
        x = embd
        for l in self.fc_layers:
            x = l(x)
            x = nn.ReLU()(x)

        prediction = self.finalLayer(x)
        return prediction.flatten()

para = {
    'epoch':90,
    'lr':0.01,
    'batch_size':2048,
    'train':0.8
}
ds = DM(rt)
print(max(ds.iId))
print(max(ds.uId))
print(len(ds.iId))
trainLen = int(para['train']*len(ds))
train,test = random_split(ds,[trainLen,len(ds)-trainLen])
dl = DataLoader(train,batch_size=para['batch_size'],shuffle=True,pin_memory=True)

model = NCF(userNum,itemNum,64, layers=[128,64,32,16,8]).to(device)
# model = SVD(userNum,itemNum,50).to(device)
optim = Adam(model.parameters(), lr=para['lr'],weight_decay=0.001)
lossfn = MSELoss()

for i in range(para['epoch']):
    for id,batch in enumerate(dl):
        print('epoch:',i,' batch:',id)
        # print(len(batch[0]))
        optim.zero_grad()

        prediction = model(batch[0].to(device), batch[1].to(device))
        loss = lossfn(batch[2].float().to(device),prediction)
        loss.backward()
        optim.step()
        print(loss)


testdl = DataLoader(test,batch_size=len(test),)
for data in testdl:
    prediction = model(data[0].to(device),data[1].to(device))
    loss = lossfn(data[2].float().to(device),prediction)
    print(loss) # MSEloss