import random
import time

from torch.utils.data import Dataset, DataLoader

from action.classifier import *
import torch
import torch.nn as nn
import torch.optim as optim
import os

idx2str = ['0m', '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
           '0p', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
           '0s', '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
           '1z', '2z', '3z', '4z', '5z', '6z', '7z']

pattern_path = 'D:/github_workspace/codes/majsoul/action/test_imgs'
datas = []

for i in range(10):
    files = os.listdir(pattern_path + '/{}m'.format(i))
    for file in files:
        file_path = pattern_path + '/{}m'.format(i) + '/' + file
        img = cv2.imread(file_path)
        datas.append((img, idx2str[i]))

    files = os.listdir(pattern_path + '/{}p'.format(i))
    for file in files:
        file_path = pattern_path + '/{}p'.format(i) + '/' + file
        img = cv2.imread(file_path)
        datas.append((img, idx2str[10 + i]))

    files = os.listdir(pattern_path + '/{}s'.format(i))
    for file in files:
        file_path = pattern_path + '/{}s'.format(i) + '/' + file
        img = cv2.imread(file_path)
        datas.append((img, idx2str[20 + i]))

for i in range(7):
    files = os.listdir(pattern_path + '/{}z'.format(i + 1))
    for file in files:
        file_path = pattern_path + '/{}z'.format(i + 1) + '/' + file
        img = cv2.imread(file_path)
        datas.append((img, idx2str[30 + i]))

classifier = Classify()

correct = 0
total = len(datas)
for data, label in datas:
    output = classifier(data)
    if label == output[0]:
        correct += 1
    else:
        print(output[0])
        cv_show(data)
print('test_imgs accuracy: {} / {} = {}'.format(correct, total, correct / total))

