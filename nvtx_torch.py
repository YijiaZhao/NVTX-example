from torch.nn import init
import torch.nn as nn
import math
import time
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torchvision
import torch.cuda.nvtx as nvtx

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50) #4*4*20
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        nvtx.range_push("fc1")
        x = F.relu(self.fc1(x))
        nvtx.range_pop()
        x = F.dropout(x, training=self.training)
        nvtx.range_push("fc2")
        x = self.fc2(x)
        nvtx.range_pop()
        return F.log_softmax(x,dim=1)

batch_size=512
epoch=20
model=Net()
for layer in model.modules():
    if isinstance(layer,nn.Linear):
        init.xavier_uniform_(layer.weight)
optimizer=torch.optim.Adam(model.parameters(),0.001)
criterion=nn.CrossEntropyLoss()
loss_holder=[]
loss_value=np.inf
step=0

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=False)


model.cuda()
criterion.cuda()
nvtx.range_push("Batch 0")
nvtx.range_push("Load Data")

for i in range(epoch):
    for batch_idx, (data,label) in enumerate(train_loader):
        nvtx.range_pop(); nvtx.range_push("Forward")
        #输出值
        data = data.cuda()
        label = label.cuda()
        outputs = model(data)
        nvtx.range_pop(); nvtx.range_push("Calculate Loss/Sync")
        #损失值
        loss = criterion(outputs, label)
        #反向传播，将所有梯度的参数清0，否则该步的梯度会和前面已经计算的梯度累乘
        optimizer.zero_grad()
        nvtx.range_pop(); nvtx.range_push("Backward")
        loss.backward()
        nvtx.range_pop(); nvtx.range_push("SGD")
        optimizer.step()
        nvtx.range_pop(); nvtx.range_pop()
        nvtx.range_push("Batch " + str(batch_idx+1)); nvtx.range_push("Load Data")
        #记录误差
        print('epoch{},Train loss{:.6f},Dealed/Records:{}/{}'.format(i,loss/batch_size,(batch_idx+1)*batch_size,60000))
        # if batch_idx%20==0:
        #     step+=1
        #     loss_holder.append([step,loss/batch_size])
        # #模型性能有所提升则保存模型，并更新loss_value
        # if batch_idx%20==0 and loss<loss_value:
        #     torch.save(model,'model.ckpt')
        #     loss_value=loss
        if batch_idx == 5:
            break
    break

nvtx.range_pop()
nvtx.range_pop()
