#!/usr/bin/env python
#-*- coding:utf-8 -*-  
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: run.py 
# @time: 2021/04/29
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
from Data import *
from GraphModel import *
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from MatrixFct import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


rt = load_data()
userNum = rt['reviewerID'].max() + 1
itemNum = rt['asin'].max() + 1

para = {
    'epoch':90,
    'lr':0.01,
    'batch_size':2048,
    'train':0.8
}
ds = DM(rt)

trainLen = int(para['train']*len(ds))
train,test = random_split(ds,[trainLen,len(ds)-trainLen])
dl = DataLoader(train,batch_size=para['batch_size'],shuffle=True,pin_memory=True)

model = GCF(userNum,itemNum, ds, 100, layers=[100,100,]).to(device)
# model = biasSVD(userNum,itemNum,50).to(device)
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
    print(loss)
