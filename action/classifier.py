# -*- coding: utf-8 -*-
# Use convolutional neural network to classify tile image
import os
import random
import time

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# CNN输出(int)与牌名(str)的对应关系
from utils import *

classes = {
    0: '0m',
    1: '1m',
    2: '2m',
    3: '3m',
    4: '4m',
    5: '5m',
    6: '6m',
    7: '7m',
    8: '8m',
    9: '9m',

    10: '0p',
    11: '1p',
    12: '2p',
    13: '3p',
    14: '4p',
    15: '5p',
    16: '6p',
    17: '7p',
    18: '8p',
    19: '9p',

    20: '0s',
    21: '1s',
    22: '2s',
    23: '3s',
    24: '4s',
    25: '5s',
    26: '6s',
    27: '7s',
    28: '8s',
    29: '9s',

    30: '1z',
    31: '2z',
    32: '3z',
    33: '4z',
    34: '5z',
    35: '6z',
    36: '7z',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def CV2PIL(img):
    return Image.fromarray(img)


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform2 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# DEBUG = True
DEBUG = False
SAVE_CLASSIFY_RES = False


class TileNet(nn.Module):
    def __init__(self):
        super(TileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, (3, 3))
        self.conv2 = nn.Conv2d(3, 16, (3, 3))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, (3, 3))
        self.conv4 = nn.Conv2d(16, 26, (3, 3))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(26 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 37)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(self.conv1(x))))
        x = self.pool2(F.relu(self.conv4(self.conv3(x))))
        x = x.view(-1, 26 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classify:

    def __init__(self):
        self.model = TileNet()
        self.model2 = TileNet()
        path = 'D:/github_workspace/codes/majsoul/weights/weight20.model1'
        path2 = 'D:/github_workspace/codes/majsoul/weights/weight20.model2'
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model2.load_state_dict(torch.load(path2, map_location=torch.device('cpu')))
        self.model.to(device)
        self.model2.to(device)
        self.__call__(np.ones((32, 32, 3), dtype=np.uint8))  # load cache

    def __call__(self, img: np.ndarray):
        tmp = img.copy()
        img = transform2(CV2PIL(img))
        img = img.view(1, *img.shape).to(device)
        with torch.no_grad():
            res = self.model(img)
            _, predicted = torch.max(res, 1)
            TileID = predicted[0]
            TileName = classes[TileID.item()]
            if DEBUG:
                print(res)
                print(TileName)
                cv_show(tmp)
            if SAVE_CLASSIFY_RES:
                cv2.imwrite('D:/github_workspace/codes/majsoul/action/classified_imgs/{}_{}.png'
                            .format(TileName, time.time()), tmp)
            if TileName in m_dict:
                h, w = tmp.shape[:2]
                img2 = tmp[:h // 2, :w]
                img2 = cv2.resize(img2, (w, h))
                tmp = img2.copy()
                img2 = transform2(CV2PIL(img2))
                img2 = img2.view(1, *img2.shape).to(device)
                res = self.model2(img2)
                _, predicted = torch.max(res, 1)
                TileID = predicted[0]
                TileName = classes[TileID.item()]
                if DEBUG:
                    print(res)
                    print(TileName)
                    cv_show(tmp)
                if SAVE_CLASSIFY_RES:
                    cv2.imwrite('D:/github_workspace/codes/majsoul/action/classified_imgs/{}_{}.png'
                                .format(TileName, time.time()), tmp)
        return TileName, res
