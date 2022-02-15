from action.classifier import *
import torch
import torch.nn as nn
import torch.optim as optim
import os

import random
import time

from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import cv2
from PIL import Image

# tiles = cv2.imread('/mnt/tiles.png')
# tiles = cv2.GaussianBlur(tiles, (9, 9), 1)
# doras = cv2.imread('/mnt/doras.png')
# doras = cv2.GaussianBlur(doras, (9, 9), 1)

model_path = 'D:/github_workspace/codes/majsoul/weights'
tiles = cv2.imread('D:/github_workspace/codes/majsoul/action/template/tiles.png')
tiles = cv2.GaussianBlur(tiles, (3, 3), 1)
doras = cv2.imread('D:/github_workspace/codes/majsoul/action/template/doras.png')
doras = cv2.GaussianBlur(doras, (3, 3), 1)

m_dict = ['0m', '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m']



w1, h1 = 72, 105
tile_locs = [
    ((323, 315), '1m'), ((401, 315), '2m'), ((478, 315), '3m'), ((555, 315), '4m'),
    ((633, 315), '5m'), ((711, 315), '6m'), ((789, 315), '7m'), ((866, 315), '8m'), ((944, 315), '9m'),
    ((322, 556), '1p'), ((400, 556), '2p'), ((477, 556), '3p'), ((555, 556), '4p'),
    ((633, 556), '5p'), ((711, 556), '6p'), ((789, 556), '7p'), ((867, 556), '8p'), ((944, 556), '9p'),
    ((322, 801), '1s'), ((400, 801), '2s'), ((478, 801), '3s'), ((556, 801), '4s'),
    ((634, 801), '5s'), ((712, 801), '6s'), ((790, 801), '7s'), ((868, 801), '8s'), ((946, 801), '9s'),
    ((1178, 314), '1z'), ((1256, 314), '2z'), ((1334, 314), '3z'), ((1412, 314), '4z'),
    ((1178, 556), '5z'), ((1256, 556), '6z'), ((1334, 556), '7z'),
]

w2, h2 = 61, 88
dora_locs = [
    ((39, 35), '0m'), ((121, 35), '0p'), ((202, 35), '0s'),
]

def add_img(data, img, name):
    PIL = Image.fromarray(img)
    img = transform(PIL)
    data.append((img, name2id[name]))


class MyDataSet1(Dataset):
    def __init__(self):
        self.data = []

        print('tiles start')

        for loc, name in tile_locs:
            print('tiles: {}'.format(name))
            x, y = loc
            img = tiles[y:y + h1, x:x + w1, :]
            img2 = cv2.resize(img, (w1, h1 // 2))
            for i in range(1):
                add_img(self.data, img, name)
                add_img(self.data, img2, name)
        print('doras start')
        for loc, name in dora_locs:
            print('tiles: {}'.format(name))
            x, y = loc
            img = doras[y:y + h2, x:x + w2, :]
            img2 = cv2.resize(img, (w2, h2 // 2))
            for i in range(800):
                add_img(self.data, img, name)
                add_img(self.data, img2, name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


class MyDataSet2(Dataset):
    def __init__(self):
        self.data = []

        print('tiles start')
        for loc, name in tile_locs:
            if name in m_dict:
                print('tiles: {}'.format(name))
                x, y = loc
                img = tiles[y:y + h1//2, x:x + w1, :]
                for i in range(800):
                    add_img(self.data, img, name)

        print('doras start')
        for loc, name in dora_locs:
            if name in m_dict:
                print('tiles: {}'.format(name))
                x, y = loc
                img = doras[y:y + h2 // 2, x:x + w2, :]
                for i in range(800):
                    cv_show(img)
                    add_img(self.data, img, name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


print('data load start')
data_set1 = MyDataSet1()
data_set2 = MyDataSet2()
print('data load finish')
print('img1 :', len(data_set1.data))
# print('img2 :', len(data_set2.data))
train_loader1 = DataLoader(dataset=data_set1, batch_size=2, shuffle=True, )
train_loader2 = DataLoader(dataset=data_set2, batch_size=2, shuffle=True, )
print('train_loader ready')

net1 = TileNet()
net2 = TileNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net1.to(device)
net2.to(device)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adamax(net1.parameters(), lr=0.001, )
optimizer2 = optim.Adamax(net2.parameters(), lr=0.001, )

# net1.load_state_dict(torch.load('/mnt/weight10.model1'))
# net2.load_state_dict(torch.load('/mnt/weight40.model2'))
epochs = 20
print('train start')

start = time.time()
for epoch in range(1, epochs + 1):
    for i, data in enumerate(train_loader1):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer1.zero_grad()
        outputs = net1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer1.step()

        if i % 400 == 0:
            print('current iter: {} / {}, current index: {} / {}, loss = {}  -> time = {}'.format(epoch, epochs, i,
                                                                                                  len(train_loader1),
                                                                                                  loss,
                                                                                                  time.time() - start))
    if epoch % 5 == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_loader1:
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = net1(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('epoch:{}, train_imgs accurency: {} / {} = {}'.format(epoch, correct, total, correct / total))
        torch.save(net1.state_dict(), '/mnt/weight{}.model1'.format(epoch))

for epoch in range(1, epochs + 1):
    for i, data in enumerate(train_loader2):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer2.zero_grad()
        outputs = net2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()

        if i % 400 == 0:
            print('current iter: {} / {}, current index: {} / {}, loss = {}  -> time = {}'.format(epoch, epochs, i,
                                                                                                  len(train_loader2),
                                                                                                  loss,
                                                                                                  time.time() - start))
    if epoch % 5 == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_loader2:
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = net2(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('epoch:{}, train_imgs accurency: {} / {} = {}'.format(epoch, correct, total, correct / total))
        torch.save(net2.state_dict(), '/mnt/weight{}.model2'.format(epoch))
print('img train finished.')