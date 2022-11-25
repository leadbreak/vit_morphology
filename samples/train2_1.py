# Setup

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from customModel import *

device = 'cuda:1'
target = 0

inputs = np.load('./inputs2.npy')

inputs[:, 0] = inputs[:, 0].astype('uint8') / 255
inputs[:, 1] = (inputs[:, 1] * 255).astype('uint8') / 255

outputs = np.load('./outputs2.npy')
outputs = outputs[:, target:target+2]

print(inputs.shape, outputs.shape)

# Build Model

num_classes = 2
batch_size, channel, height, width = inputs.shape
x = torch.Tensor(inputs[:1])
vit = ViT(in_channels=channel, num_classes=num_classes)


# custom dataset
dataset = customDataset(X=inputs, Y=outputs)
batch_size = 5
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# test dataloader
batch_iterator = iter(dataloader)
inputs, labels = next(batch_iterator)
print(inputs.size(), labels.size())

def train_model(net, dataloader, criterion, optimizer, num_epochs) :
    global device
    
    minimumLoss = 1
    device = torch.device(device)
    print("사용장치 :", device)
    
    # 네트워크를 device로
    net.to(device)
    
    # 네트워크 가속화
    torch.backends.cudnn.benchmark = True
    
    # Training
    for epoch in range(num_epochs) :
        
        # epoch 별 훈련 및 검증 루프
        for phase in ['train', 'val'] :
            if phase == 'trian' :
                net.train()
            else :
                net.eval()
            
            epoch_loss = 0.0   # epoch 손실 합
            epoch_corrects = 0 # epoch 정답 수
            
            # 학습하지 않았을 때의 검증 성능을 확인하기 위해 epoch=0의 훈련 생략
            if (epoch == 0) and (phase=='train') :
                continue
            
            # if (epoch % 10 == 9) & (phase=='train') :
            print()
            print(f"Epoch {epoch+1}/{num_epochs}", end=' ')
            
            # 데이터 로더에서 미니 배치를 꺼내 루프
            for inputs, labels in dataloader :
                
                # GPU를 사용할 수 있으면 GPU에 데이터 보냄
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # reset optimizer
                optimizer.zero_grad()
                
                # calculate forward propagation
                with torch.set_grad_enabled(phase=='train') :
                    outputs = net(inputs)
                    
                    loss = criterion(outputs, labels)
                    
                    # 훈련 시에는 오차 역전파
                    if phase == 'train' :
                        loss.backward()
                        optimizer.step()
                        
                    # 결과 계산
                    epoch_loss += loss.item() * inputs.size(0) # 손실합계 갱신
                    # epoch_corrects += torch.sum(outputs == labels.data)
                        
            # epoch 별 손실과 정답률 표시
            epoch_loss = epoch_loss / len(dataloader.dataset)


            print(f"{phase} Loss : {epoch_loss:.4f}", end=' ')
            if phase=='train' :
                t_loss = epoch_loss
            elif phase=='val' :
                v_loss = epoch_loss
                if (epoch > 0) & (v_loss < minimumLoss) :
                    minimumLoss = v_loss
                    fileName = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f"_{epoch+1}"+ f"_{t_loss:.4f}" + f"_{v_loss:.4f}.pt" # 생성 시간과 개수로 저장
                    torch.save(net, f"./models/label_{target}/{fileName}")
                    print(f"- Model Saved", end=' ')

vit.train()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(vit.parameters(), lr=1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_epochs=30

train_model(vit, dataloader, criterion, optimizer, num_epochs) 

vit.train()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(vit.parameters(), lr=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_epochs=100

train_model(vit, dataloader, criterion, optimizer, num_epochs) 

vit.train()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(vit.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_epochs=100

train_model(vit, dataloader, criterion, optimizer, num_epochs) 

vit.train()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(vit.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_epochs=300

train_model(vit, dataloader, criterion, optimizer, num_epochs) 

vit.train()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(vit.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

num_epochs=1000

train_model(vit, dataloader, criterion, optimizer, num_epochs) 