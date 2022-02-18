# -*- coding: utf-8 -*-
# 获取屏幕信息，并通过视觉方法标定手牌与按钮位置，仿真鼠标点击操作输出
import os
import random
import time
from typing import List, Tuple

import cv2
import pyautogui
import numpy as np
from PIL import Image

from strategy import ScreenInfo, Strategy
from utils import *

from .classifier import Classify, transform

pyautogui.PAUSE = 0  # 函数执行后暂停时间
pyautogui.FAILSAFE = True  # 开启鼠标移动到左上角自动退出

DEBUG = False  # 是否显示检测中间结果
TIME_LOG = False

REAL_MODE = True


def screenShot():
    if REAL_MODE:
        img = np.asarray(pyautogui.screenshot())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        img = cv2.imread('action/scenes/scene_chi.png')
        return img


def PosTransform(pos, M: np.ndarray) -> np.ndarray:
    return np.int32(cv2.perspectiveTransform(np.float32([[pos]]), M)[0][0])


def Similarity(img1: np.ndarray, img2: np.ndarray):
    assert (len(img1.shape) == len(img2.shape) == 3)
    if img1.shape[0] < img2.shape[0]:
        img1, img2 = img2, img1
    n, m, c = img2.shape
    img1 = cv2.resize(img1, (m, n))  # 大图缩小
    if DEBUG:
        cv2.imshow('diff', np.uint8(np.abs(np.float32(img1) - np.float32(img2))))
        cv2.waitKey(1)
    ksize = max(1, min(n, m) // 50)
    img1 = cv2.blur(img1, ksize=(ksize, ksize))
    img2 = cv2.blur(img2, ksize=(ksize, ksize))
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    if DEBUG:
        cv2.imshow('bit', np.uint8((np.abs(img1 - img2) < 30).sum(2) == 3) * 255)
        cv2.waitKey(1)
    return ((np.abs(img1 - img2) < 30).sum(2) == 3).sum() / (n * m)


def ObjectLocalization(objImg: np.ndarray, targetImg: np.ndarray) -> np.ndarray:
    """
    https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    Feature based object detection
    return: Homography matrix M (objImg->targetImg), if not found return None
    """
    img1 = objImg
    img2 = targetImg
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=5000)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # store all the good matches as per Lowe's ratio test.
    good = []
    for i, j in enumerate(matches):
        if len(j) == 2:
            m, n = j
            if m.distance < 0.7 * n.distance:
                good.append(m)
                matchesMask[i] = [1, 0]
    print('Number of good matches:', len(good))
    if DEBUG:
        # draw
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, matches, None, **draw_params)
        img3 = cv2.pyrDown(img3)
        cv2.imshow('ORB match', img3)
        cv2.waitKey(1)
    # Homography
    MIN_MATCH_COUNT = 50
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if DEBUG:
            # draw
            matchesMask = mask.ravel().tolist()
            h, w, d = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(img2, [np.int32(dst)],
                                 True, (0, 0, 255), 10, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                                   good, None, **draw_params)
            img3 = cv2.pyrDown(img3)
            cv2.imshow('Homography match', img3)
            cv2.waitKey(1)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        M = None
    assert (type(M) == type(None) or (
            type(M) == np.ndarray and M.shape == (3, 3)))
    return M


def getHomographyMatrix(img1, img2, threshold=0.3):
    # if similarity>threshold return M
    # else return None
    M = ObjectLocalization(img1, img2)
    if M is not None:
        print('Homography Matrix:', M)
        n, m, c = img1.shape
        x0, y0 = np.int32(PosTransform([0, 0], M))
        x1, y1 = np.int32(PosTransform([m, n], M))
        sub_img = img2[y0:y1, x0:x1, :]
        S = Similarity(img1, sub_img)
        print('Similarity:', S)
        if S > threshold:
            return M
    return None


class Layout:
    size = (1920, 1080)  # 界面长宽
    duanWeiChang = (1348, 321)  # 段位场按钮
    menuButtons = [(1382, 406), (1382, 573), (1382, 740),
                   (1383, 885), (1393, 813)]  # 铜/银/金之间按钮
    tileSize = (95, 152)  # 自己牌的大小
    tileSize_paihe = (37, 70)  # 牌河牌的大小
    tileSize_dora = (51, 78)  # dora牌的大小
    tileSize_fulu = (40, 0)  # 副露牌的大小


class GUIInterface:

    def __init__(self):
        # 类中数值均为(1920, 1080)的标准大小下的坐标，需用M转换为实际坐标
        if TIME_LOG:
            self.start_time = time.time()
        self.M = None  # Homography matrix from (1920,1080) to real window
        self.waitPos = [1800, 780]
        # load template imgs
        join = os.path.join
        root = os.path.dirname(__file__)

        def load(name):
            return cv2.imread(join(root, 'template', name))

        self.screenImg0 = None
        self.screenImg = None
        self.screenImg2 = None
        self.selfWind = None
        self.roundWind = None
        self.yiPai = None
        self.menuImg = load('menu.png')  # 初始菜单界面
        self.duanweichangImg = load('duanweichang.png')
        self.yinzhijianImg = load('yinzhijian.png')
        self.jinzhijianImg = load('jinzhijian.png')
        self.sirennan = load('sirennan.png')
        self.queding = load('queding.png')
        self.queding2 = load('queding2.png')
        self.chiImg = load('chi.png')
        self.pengImg = load('peng.png')
        self.gangImg = load('gang.png')
        self.huImg = load('hu.png')
        self.zimoImg = load('zimo.png')
        self.tiaoguoImg = load('tiaoguo.png')
        self.liqiImg = load('liqi.png')
        self.ButtonImgs = [(self.huImg, 'hu'), (self.zimoImg, 'zimo'),
                           (self.liqiImg, 'liqi'),
                           (self.chiImg, 'chi'),
                           (self.pengImg, 'peng'),
                           # (self.gangImg, 'gang'),
                           ]
        self.selfWindPatterns = [(load('east.png'), 0), (load('south.png'), 1),
                                 (load('west.png'), 2), (load('north.png'), 3)]
        self.roundWindPatterns = [(load('east2.png'), 0), (load('south2.png'), 1), (load('west2.png'), 2)]

        if TIME_LOG:
            img_load = time.time()
            print('img load finish:', img_load - self.start_time)
        self.paihe_buf = None
        self.fulu_buf = None
        self.handTiles = None
        self.doraTiles = None
        self.paihe_buf = None
        self.fulu_buf = None
        self.paiHeTiles = None
        self.fuLuTiles = None
        # load classify model
        self.classify = Classify()
        self.sceenInfo = None
        self.method = None

    def calibrateMenu(self):
        # self.M = getHomographyMatrix(self.menuImg, screenShot(), threshold=0.7)
        self.M = np.array([[7.22243006e-01, 7.10414950e-05, 3.02689377e+01],
                           [2.05461098e-04, 7.21891351e-01, 2.11524639e+01],
                           [9.51698899e-07, 3.84192517e-08, 1.00000000e+00]])
        if self.M is not None:
            self.waitPos = self.PosTransform(self.waitPos)
            return True
        return False

    def PosTransform(self, pos):
        return PosTransform(pos, self.M)

    def click(self, x, y):
        # x, y = self.PosTransform([x, y])
        pyautogui.FAILSAFE = False
        pyautogui.moveTo(x=x, y=y)
        pyautogui.click(x=x, y=y, button='left', duration=0.2)
        time.sleep(0.1)
        # time.sleep(1)
        pyautogui.moveTo(x=self.waitPos[0], y=self.waitPos[1])

    def screenShot(self):
        # img = screenShot()
        # x0, y0 = np.int32(self.PosTransform([0, 0]))
        # x1, y1 = np.int32(self.PosTransform(Layout.size))
        # img = img[y0 - 10:y1 + 3, x0 - 20:x1 + 30, :]
        # # cv_show(img)
        # return cv2.resize(img, Layout.size)
        return screenShot()

    def flush_screen_img(self):
        screen_img1 = self.screenShot()
        time.sleep(0.3)
        screen_img2 = self.screenShot()
        self.screenImg0 = np.array(screen_img2)  # 原始画面
        self.screenImg = np.minimum(screen_img1, screen_img2)  # 消除高光动画
        self.screenImg2 = np.maximum(screen_img1, screen_img2)  # 增强高光动画

    def flush(self, click=False, all=True):
        if TIME_LOG:
            flush_begin = time.time()
            print('flush_begin finish:', flush_begin - self.start_time)
        self.flush_screen_img()
        if click:
            pyautogui.click(x=self.waitPos[0], y=self.waitPos[1], button='left', duration=0.2)
            if TIME_LOG:
                preClick_s = time.time()
                print('preClick_s finish:', preClick_s - self.start_time)
            self.preClick()
            if TIME_LOG:
                preClick_e = time.time()
                print('preClick_e finish:', preClick_e - self.start_time)

        if TIME_LOG:
            screen_flush = time.time()
            print('screen_flush finish:', screen_flush - self.start_time)

        if not all:
            return

        self.selfWind = self.getSelfWind()
        self.roundWind = self.getRoundWind()
        self.yiPai = sanyuan_dict + [self.roundWind, self.selfWind]
        self.handTiles = self.getHandTiles()
        self.doraTiles = self.getDoras()
        self.paihe_buf = (self.getMyPaiHe(), self.getXiaJiaPaiHe(),
                          self.getDuiJiaPaiHe(), self.getShangJiaPaiHe())
        self.fulu_buf = (self.getMyFuLu(), self.getXiaJiaFuLu(),
                         self.getDuiJiaFuLu(), self.getShangJiaFuLu())
        self.paiHeTiles = self.paihe_buf[1] + self.paihe_buf[2] + self.paihe_buf[3]
        self.fuLuTiles = self.fulu_buf[1] + self.fulu_buf[2] + self.fulu_buf[3]

        self.sceenInfo = ScreenInfo(self.roundWind, self.selfWind, self.doraTiles, self.handTiles,
                                    self.paihe_buf[0], self.paihe_buf[1], self.paihe_buf[2], self.paihe_buf[3],
                                    self.fulu_buf[0], self.fulu_buf[1], self.fulu_buf[2], self.fulu_buf[3],
                                    self.yiPai)
        self.method = Strategy(self.sceenInfo)

        if TIME_LOG:
            screen_info = time.time()
            print('screen_info finish:', screen_info - self.start_time)

    def run(self):
        # self.actionChiPeng([1, 1], [1, 1])
        if TIME_LOG:
            run_start = time.time()
            print('run_start finish:', run_start - self.start_time)
        clicked = self.autoButtonClick()
        if TIME_LOG:
            autoButtonClick_time = time.time()
            print('autoButtonClick finish:', autoButtonClick_time - self.start_time)
        if len(self.handTiles) % 3 == 2 and len(self.handTiles + self.fulu_buf[0]) >= 14 \
                and not clicked and len(self.handTiles) < 15:
            self.actionZimo()  # 防止空隙时间内出现自摸直接进入舍牌
            if len(self.handTiles) > 0:
                print('new tile:', self.handTiles[-1][0])
            tile, op, shantin, liqi = self.method.getNextAction()
            print(
                'tile=\033[0;33m{}\033[0m, shantin=\033[0;33m{}\033[0m, liqi=\033[0;33m{}\033[0m'.format(tile, shantin,
                                                                                                         liqi))
            print()
            if TIME_LOG:
                liqi_s = time.time()
                print('liqi_s finish:', liqi_s - self.start_time)
            if liqi:
                print('liqi')
                time.sleep(1)
                self.actionLiqi()
            if TIME_LOG:
                liqi_e = time.time()
                print('liqi_e finish:', liqi_e - self.start_time)
            # time.sleep(random.uniform(0, 0.8))
            self.actionDiscardTile(tile)
            if TIME_LOG:
                actionDiscardTile_t = time.time()
                print('actionDiscardTile finish:', actionDiscardTile_t - self.start_time)

        if TIME_LOG:
            discard = time.time()
            print('discard finish:', discard - self.start_time)

    def actionDiscardTile(self, tile: str):
        self.flush_screen_img()
        self.handTiles = self.getHandTiles()
        for t, (x, y) in self.handTiles:
            if t == tile:
                self.click(x, y)
                return True
        # 如果是5m/p/s而手牌没有，则打0m/p/s
        if tile == '5m':
            self.actionDiscardTile('0m')
            return True
        if tile == '5p':
            self.actionDiscardTile('0p')
            return True
        if tile == '5s':
            self.actionDiscardTile('0s')
            return True
        return False
        # raise Exception(
        #     'GUIInterface.discardTile tile not found. L:', L, 'tile:', tile)

    # 点击“段位场”、“银之间”和“四人南”按钮
    def preClick(self):
        n, m, _ = self.duanweichangImg.shape
        x0, y0 = 1000, 192
        x1, y1 = 1700, 912
        img = self.screenImg[y0:y1, x0:x1, :]
        T = cv2.matchTemplate(img, self.duanweichangImg, cv2.TM_SQDIFF, mask=self.duanweichangImg.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        dst = img[y:y + n, x:x + m].copy()
        dst[self.duanweichangImg == 0] = 0
        if Similarity(self.duanweichangImg, dst) >= 0.7:
            x, y = x + x0 + m // 2, y + y0 + n // 2
            self.click(x, y)

        n, m, _ = self.yinzhijianImg.shape
        img = self.screenImg[y0:y1, x0:x1, :]
        T = cv2.matchTemplate(img, self.yinzhijianImg, cv2.TM_SQDIFF, mask=self.yinzhijianImg.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        dst = img[y:y + n, x:x + m].copy()
        dst[self.yinzhijianImg == 0] = 0
        if Similarity(self.yinzhijianImg, dst) >= 0.8:
            x, y = x + x0 + m // 2, y + y0 + n // 2
            self.click(x, y)

        n, m, _ = self.sirennan.shape
        img = self.screenImg[y0:y1, x0:x1, :]
        T = cv2.matchTemplate(img, self.sirennan, cv2.TM_SQDIFF, mask=self.sirennan.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        dst = img[y:y + n, x:x + m].copy()
        dst[self.sirennan == 0] = 0
        if Similarity(self.sirennan, dst) >= 0.8:
            x, y = x + x0 + m // 2, y + y0 + n // 2
            self.click(x, y)
            time.sleep(10)

        n, m, _ = self.queding.shape
        x0, y0 = 1660, 892
        x1, y1 = 1920, 1022
        img = self.screenImg[y0:y1, x0:x1, :]
        T = cv2.matchTemplate(img, self.queding, cv2.TM_SQDIFF, mask=self.queding.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        dst = img[y:y + n, x:x + m].copy()
        dst[self.queding == 0] = 0
        if Similarity(self.queding, dst) >= 0.8:
            x, y = x + x0 + m // 2, y + y0 + n // 2
            self.click(x, y)

        n, m, _ = self.queding2.shape
        x0, y0 = 1660, 892
        x1, y1 = 1920, 1022
        img = self.screenImg[y0:y1, x0:x1, :]
        T = cv2.matchTemplate(img, self.queding2, cv2.TM_SQDIFF, mask=self.queding2.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        dst = img[y:y + n, x:x + m].copy()
        dst[self.queding2 == 0] = 0
        if Similarity(self.queding2, dst) >= 0.8:
            x, y = x + x0 + m // 2, y + y0 + n // 2
            self.click(x, y)

    def deref_dora(self, tile):
        if tile == '0m':
            return '5m'
        if tile == '0p':
            return '5p'
        if tile == '0s':
            return '5s'
        return tile

    def cast_dora(self, tile):
        handTiles = [tile[0] for tile in self.handTiles]
        if tile == '5m' and '5m' not in handTiles:
            return '0m'
        if tile == '5p' and '5p' not in handTiles:
            return '0p'
        if tile == '5s' and '5s' not in handTiles:
            return '0s'
        return tile

    def actionChiPeng(self, chiLoc, pengLoc):
        turn = self.getTurn()
        if len(self.paihe_buf[turn + 1]) == 0:
            return False
        if len(self.paihe_buf[turn + 1][0] + self.paihe_buf[turn + 1][1]) == 0:
            print('chi/peng empty')
            return False
        lastDiscardTile = (self.paihe_buf[turn + 1][0] + self.paihe_buf[turn + 1][1])[-1]
        lastDiscardTile = self.deref_dora(lastDiscardTile[0])

        print('\033[0;36mchi/peng available: \033[0m: \033[0;31m{}\033[0m -> {}'.format(turn, lastDiscardTile))

        if turn == 2:
            ops = [Operation.NoEffect, Operation.Chi, Operation.Peng]
        else:
            ops = [Operation.NoEffect, Operation.Peng]
        tile, op, choice, liqi = self.method.getNextAction(lastDiscardTile, ops)
        if tile is not None:
            if choice is not None:  # 如果是None则表明强制弃和防守
                print('tile=\033[0;33m{}\033[0m, choice=\033[0;33m{}\033[0m, liqi=\033[0;33m{}\033[0m'.format(tile,
                                                                                                              choice,
                                                                                                              liqi))
            else:
                print('\033[0;33mdon\'t chi/peng due to defense\033[0m')
                return False
        else:
            print('\033[0;33mdon\'t chi/peng\033[0m')
            return False
        print()

        time.sleep(0.4)
        self.screenImg = self.screenShot()
        if op == Operation.NoEffect:
            return False
        if op == Operation.Chi:  # 只能吃上家的牌
            choice = [self.cast_dora(ch) for ch in choice]
            if chiLoc is not None:
                self.click(*chiLoc)
                time.sleep(1)
                self.screenImg = self.screenShot()
                self.clickCandidateMeld(choice)
                return True
            return False
        elif op == Operation.Peng:
            if pengLoc is not None:
                self.click(*pengLoc)
        time.sleep(0.8)
        self.actionDiscardTile(tile)
        return True

    def actionZimo(self):
        x0, y0 = 595, 557
        x1, y1 = 1508, 912
        img = self.screenImg0[y0:y1, x0:x1, :]
        loc = self.existButton(img, x0, y0, self.zimoImg)
        if loc is None:
            return
        x, y = loc
        self.click(x, y)

    def autoButtonClick(self):
        # 优先级：自摸 = 荣和 > 立直 > 碰/杠/吃 > 跳过
        x0, y0 = 595, 557
        x1, y1 = 1508, 912
        img = self.screenImg0[y0:y1, x0:x1, :]
        tiaoguo_loc = self.existButton(img, x0, y0, self.tiaoguoImg)
        if tiaoguo_loc is None:  # 若是不存在跳过按钮，则其他的就不用找了
            return False

        chiLoc = None
        pengLoc = None
        for btn_img, name in self.ButtonImgs:
            loc = self.existButton(img, x0, y0, btn_img)
            if loc is None:
                continue

            if name == 'hu' or name == 'zimo':
                x, y = loc
                self.click(x, y)
                return True
            if name == 'liqi':  # 立直交给actionLiqi()完成，必要的，忽略立直按钮则会按跳过
                print('liqi...')
                return False
            if name == 'chi':
                chiLoc = loc
            if name == 'peng':
                pengLoc = loc

        if self.actionChiPeng(chiLoc, pengLoc):
            time.sleep(1)
            # 吃碰后手牌画面会变化
            self.flush_screen_img()
            self.handTiles = self.getHandTiles()
            return True
        x, y = tiaoguo_loc
        self.click(x, y)
        return False

    def actionLiqi(self):
        # x0, y0 = 0, 450
        # x1, y1 = 80, 812
        # self.click(x0+32, y0+112)
        # img = self.screenImg[y0:y1, x0:x1, :]
        # coordinates = [(32, 112)]
        # img_cpy = img.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print('liqi start')
        x0, y0 = 595, 557
        x1, y1 = 1508, 912
        self.screenImg0 = np.array(self.screenShot())
        img = self.screenImg0[y0:y1, x0:x1, :]
        # cv_show(img)
        res = self.existButton(img, x0, y0, self.liqiImg)
        if res is None:
            print('liqi not exist')
            return False
        else:
            x, y = res
            self.click(x, y)
            print('liqi exist')
            # 点击左侧的“和”按钮
            x0, y0 = 0, 450
            x1, y1 = 80, 812
            self.click(x0 + 32, y0 + 112)
            return True

    def getHandTiles(self) -> List[Tuple[str, Tuple[int, int]]]:
        result = []
        img = self.screenImg.copy()  # for calculation
        start = [235, 1002]
        colorThreshold = 110
        tileThreshold = 0.8 * np.int32(Layout.tileSize)
        fail = 0
        maxFail = 100
        i = 0
        while fail < maxFail:
            x, y = start[0] + i, start[1]
            if x >= 1920:
                break
            if all(img[y, x, :] > colorThreshold):
                fail = 0
                img[y, x, :] = colorThreshold
                retval, image, mask, rect = cv2.floodFill(
                    image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                    loDiff=(0, 0, 0), upDiff=tuple([255 - colorThreshold] * 3), flags=cv2.FLOODFILL_FIXED_RANGE)
                x, y, dx, dy = rect
                if dx > tileThreshold[0] and dy > tileThreshold[1]:
                    tile_img = self.screenImg[y:y + dy, x:x + dx, :]
                    tileStr, res = self.classify(tile_img)
                    # cv2.imshow('tile', tile_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    result.append((tileStr, (x + dx // 2, y + dy // 2)))
                    i = x + dx - start[0]
            else:
                fail += 1
            i += 1
        return result

    def get_image_rotation(self, image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW // 2) - cX
        M[1, 2] += (nH // 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def getMyPaiHe(self):
        x, y = 765, 530
        w, h = 420, 420
        img = self.screenImg[y:y + h, x:x + w]
        img = cv2.resize(img, (280, 280))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        coordinates = [(13, 5), (264, 4), (270, 123), (4, 122)]
        # img_cpy = img.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1 = np.array(coordinates, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (280, 0), (280, 280), (0, 280)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
        raw_img = cv2.warpPerspective(img.copy(), M, (0, 0))[:280, :280]
        # raw_img = raw_img[169:270, 0:130]
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.getPaiheResult(raw_img)

    def getShangJiaPaiHe(self):
        x, y = 510, 300
        w, h = 280, 280
        img = self.screenImg[y:y + h, x:x + w]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = self.get_image_rotation(img, 90)
        coordinates = [(3, 1), (258, 25), (258, 250), (3, 210)]
        # img_cpy = img.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1 = np.array(coordinates, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (280, 0), (280, 280), (0, 280)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
        raw_img = cv2.warpPerspective(img.copy(), M, (0, 0))[:280, :280]
        # raw_img = raw_img[169:270, 0:130]
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.getPaiheResult(raw_img)

    def getMyFuLu(self):
        x, y = 810, 600
        w, h = 960, 400
        img = self.screenImg[y:y + h, x:x + w]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        coordinates = [(14, 307), (930, 309), (953, 380), (16, 382)]
        # img_cpy = img.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1 = np.array(coordinates, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (840, 0), (840, 100), (0, 100)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
        raw_img = cv2.warpPerspective(img.copy(), M, (0, 0))[:100, :840]
        raw_img = cv2.resize(raw_img, (840, 100))
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.getFuluResult(raw_img, 1.2)

    def getShangJiaFuLu(self):
        x, y = 40, 200
        w, h = 840, 840
        img = self.screenImg[y:y + h, x:x + w]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = self.get_image_rotation(img, 90)
        coordinates = [(14, 460), (730, 674), (730, 758), (14, 520)]
        img_cpy = img.copy()
        for coor in coordinates:
            cv2.circle(img_cpy, (int(coor[0]), int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1 = np.array(coordinates, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (840, 0), (840, 100), (0, 100)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
        raw_img = cv2.warpPerspective(img.copy(), M, (0, 0))[:100, :840]
        raw_img = cv2.resize(raw_img, (840, 100))
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.getFuluResult(raw_img, 1.5)

    def getDuiJiaFuLu(self):
        x, y = 400, 10
        w, h = 840, 840
        img = self.screenImg[y:y + h, x:x + w]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = self.get_image_rotation(img, 180)
        coordinates = [(14, 750), (820, 750), (810, 784), (12, 786)]
        # img_cpy = img.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1 = np.array(coordinates, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (840, 0), (840, 100), (0, 100)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
        raw_img = cv2.warpPerspective(img.copy(), M, (0, 0))[:100, :840]
        raw_img = cv2.resize(raw_img, (840, 100))
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.getFuluResult(raw_img, 1.5)

    def getXiaJiaFuLu(self):
        x, y = 1000, 10
        w, h = 840, 840
        img = self.screenImg[y:y + h, x:x + w]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = self.get_image_rotation(img, -90)
        coordinates = [(364, 596), (825, 464), (825, 515), (362, 672)]
        # img_cpy = img.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1 = np.array(coordinates, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (840, 0), (840, 100), (0, 100)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
        raw_img = cv2.warpPerspective(img.copy(), M, (0, 0))[:100, :840]
        raw_img = cv2.resize(raw_img, (840, 100))
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.getFuluResult(raw_img, 1.5)

    def getXiaJiaPaiHe(self):
        x, y = 1115, 270
        w, h = 280, 280
        img = self.screenImg[y:y + h, x:x + w]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = self.get_image_rotation(img, -90)
        coordinates = [(27, 28), (268, 7), (268, 203), (27, 250)]
        # img_cpy = img.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1 = np.array(coordinates, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (280, 0), (280, 280), (0, 280)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
        raw_img = cv2.warpPerspective(img.copy(), M, (0, 0))[:280, :280]
        # raw_img = raw_img[169:270, 0:130]
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.getPaiheResult(raw_img)

    def getDuiJiaPaiHe(self):
        x, y = 785, 10
        w, h = 330, 330
        img = self.screenImg[y:y + h, x:x + w]
        # img = cv2.resize(img, (280, 280))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = self.get_image_rotation(img, -180)
        coordinates = [(5, 44), (320, 44), (307, 163), (15, 163)]
        # img_cpy = img.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1 = np.array(coordinates, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (280, 0), (280, 280), (0, 280)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
        raw_img = cv2.warpPerspective(img.copy(), M, (0, 0))[:280, :280]
        # raw_img = raw_img[169:270, 0:130]
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.getPaiheResult(raw_img)

    def getPaiheResult(self, raw_img):
        img = raw_img.copy()
        liqi = False  # 是否立直，通过牌河横置的牌判断
        result = []  # 立直前
        result2 = []  # 立直后
        # starts = [[4, 4], [4, 96], [4, 193]]
        starts = [[4, 39], [4, 131], [4, 228]]
        colorThreshold = 190
        tileThreshold = 0.8 * np.int32(Layout.tileSize_paihe)
        maxFail = 100

        for start in starts:
            # x, y = start
            # draw_img = raw_img[y:y + 250, x:x + 250]
            # cv2.imshow('draw_img', draw_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            fail = 0
            i = 0
            while fail < maxFail:
                x, y = start[0] + i, start[1]
                if x >= 280:
                    break
                if all(img[y, x, :] > colorThreshold):
                    fail = 0
                    img[y, x, :] = colorThreshold
                    retval, image, mask, rect = cv2.floodFill(
                        image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                        loDiff=(tuple([0] * 3)), upDiff=tuple([255 - colorThreshold] * 3),
                        flags=cv2.FLOODFILL_FIXED_RANGE)
                    x, y, dx, dy = rect
                    if dx > tileThreshold[0] and dy > tileThreshold[1]:
                        tile_img = raw_img[y:y + dy, x:x + dx, :]
                        rotated = False
                        if dy < 1.5 * dx and not liqi:  # 横置的牌，后一个条件是是保证识别出多个横置牌不会报错
                            rotated = True
                            liqi = True
                            tile_img = self.get_image_rotation(tile_img, -90)
                        tileStr, res = self.classify(tile_img)
                        # print(tileStr, liqi)
                        # cv2.imshow('tile', tile_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        if not liqi:
                            result.append((tileStr, rotated))
                        else:
                            result2.append((tileStr, rotated))
                        i = x + dx - start[0]
                else:
                    fail += 1
                i += 1
        return result, result2

    def getFuluResult(self, raw_img, ratio_threshold):  # raw_img = 100 * 840
        img = raw_img.copy()
        result = []
        start = [835, 90]
        colorThreshold = 190
        tileThreshold = 0.8 * np.int32(Layout.tileSize_fulu)

        # cv2.imshow('draw_img', draw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        i = 0
        fail = 0
        maxFail = 100
        while fail < maxFail:
            x, y = start[0] - i, start[1]
            if x <= 10:
                break
            if all(img[y, x, :] > colorThreshold):
                fail = 0
                img[y, x, :] = colorThreshold
                retval, image, mask, rect = cv2.floodFill(
                    image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                    loDiff=(tuple([0] * 3)), upDiff=tuple([255 - colorThreshold] * 3),
                    flags=cv2.FLOODFILL_FIXED_RANGE)
                x, y, dx, dy = rect
                if dx > tileThreshold[0] and dy > tileThreshold[1]:
                    tile_img = raw_img[y:y + dy, x:x + dx, :]
                    rotated = False
                    if dy < ratio_threshold * dx:  # 横置的牌
                        rotated = True
                        tile_img = self.get_image_rotation(tile_img, -90)
                    # tmp_img = cv2.resize(tile_img, (32, 32))
                    # cv2.imshow('tile', tmp_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    tileStr, res = self.classify(tile_img)
                    result.append((tileStr, rotated))
                    i = -(x + dx) + start[0]
            else:
                fail += 1
            i += 1
        return result

    # 判断此时是哪一方的出牌时间，根据是中间的黄色闪烁区域
    def getTurn(self):
        self.flush_screen_img()
        flag = [1020, 495]
        coordinates = [[1067, 420], [952, 352], [853, 420]]  # my = [950, 500],
        # img_cpy = self.screenImg.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        diffs = []
        for coor in coordinates:
            x, y = coor
            # print(self.screenImg[y-2:y+2, x-2:x+2, :], self.screenImg[flag[1]-2:flag[1]+2, flag[0]-2:flag[0]+2, :],
            # self.screenImg[flag[1], flag[0], :] - self.screenImg[y, x, :])
            img_diff = cv2.subtract(self.screenImg2[y - 2:y + 2, x - 2:x + 2, :],
                                    self.screenImg2[flag[1] - 2:flag[1] + 2, flag[0] - 2:flag[0] + 2, :])
            # print(img_diff)
            diffs.append(sum(sum(sum(img_diff))))
        return diffs.index(max(diffs))

    def getDoras(self):
        x, y = 26, 73
        w, h = 260, 78
        # ww, hh = Layout.tileSize_dora
        raw_img = self.screenImg[y:y + h, x:x + w]
        # cv2.imshow('raw_img', raw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img = raw_img.copy()
        result = []
        start = [3, 12]
        colorThreshold = 110
        tileThreshold = 0.7 * np.int32(Layout.tileSize_dora)
        maxFail = 100
        fail = 0
        i = 0
        while fail < maxFail:
            x, y = start[0] + i, start[1]
            if x >= 260:
                break
            if all(img[y, x, :] > colorThreshold):
                fail = 0
                img[y, x, :] = colorThreshold
                retval, image, mask, rect = cv2.floodFill(
                    image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                    loDiff=(tuple([0] * 3)), upDiff=tuple([255 - colorThreshold] * 3), flags=cv2.FLOODFILL_FIXED_RANGE)
                x, y, dx, dy = rect
                if dx > tileThreshold[0] and dy > tileThreshold[1]:
                    tile_img = raw_img[y:y + dy, x:x + dx, :]
                    # cv2.imshow('tile', tile_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    tileStr, res = self.classify(tile_img)
                    result.append(tileStr)
                    i = x + dx - start[0]
            else:
                fail += 1
            i += 1
        return result

    def getRoundWind(self):
        x0, y0 = 920, 380
        x1, y1 = 958, 432
        img = self.screenImg[y0:y1, x0:x1, :]
        winds = ['1z', '2z', '3z']
        sim = [0] * 3
        for wind, i in self.roundWindPatterns:
            n, m, _ = wind.shape
            T = cv2.matchTemplate(img, wind, cv2.TM_SQDIFF, mask=wind.copy())
            _, _, (x, y), _ = cv2.minMaxLoc(T)
            dst = img[y:y + n, x:x + m].copy()
            dst[wind == 0] = 0
            sim[i] = Similarity(wind, dst)
        return winds[sim.index(max(sim))]

    def getSelfWind(self):
        x0, y0 = 800, 480
        x1, y1 = 878, 532
        img = self.screenImg[y0:y1, x0:x1, :]
        winds = ['1z', '2z', '3z', '4z']
        sim = [0] * 4
        for wind, i in self.selfWindPatterns:
            n, m, _ = wind.shape
            T = cv2.matchTemplate(img, wind, cv2.TM_SQDIFF, mask=wind.copy())
            _, _, (x, y), _ = cv2.minMaxLoc(T)
            dst = img[y:y + n, x:x + m].copy()
            dst[wind == 0] = 0
            sim[i] = Similarity(wind, dst)
        return winds[sim.index(max(sim))]

    # 判断界面上是否存在buttonImg
    def existButton(self, img, x0, y0, buttonImg, similarityThreshold=0.6):
        n, m, _ = buttonImg.shape
        T = cv2.matchTemplate(img, buttonImg, cv2.TM_SQDIFF, mask=buttonImg.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        dst = img[y:y + n, x:x + m].copy()
        dst[buttonImg == 0] = 0

        if Similarity(buttonImg, dst) >= similarityThreshold:
            return x + x0 + m // 2, y + y0 + n // 2
        return None

    def clickCandidateMeld(self, tiles: List[str]):
        # 有多种不同的吃碰方法，二次点击选择
        img = self.screenImg.copy()  # for calculation
        start = [600, 720]
        end = [1400, 720]
        # coordinates = [[600, 720], [1400, 720]]
        # img_cpy = self.screenImg.copy()
        # for coor in coordinates:
        #     cv2.circle(img_cpy, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
        # cv2.imshow('img', img_cpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        leftBound = rightBound = start[0]
        colorThreshold = 180
        tileThreshold = 0.7 * np.int32((30, 30))
        matched_tiles = []
        tiles = sorted(tiles, key=lambda x: x[0])  # 升序

        print('\033[0;36mto meld: \033[0m: \033[0;31m{}\033[0m'.format(tiles))
        fail = 0
        maxFail = 400
        i = 0
        while 1:
            x, y = start[0] + i, start[1]
            if x >= end[0]:
                break
            if all(img[y, x, :] > colorThreshold):
                fail = 0
                img[y, x, :] = colorThreshold
                retval, image, mask, rect = cv2.floodFill(
                    image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                    loDiff=(tuple([0] * 3)), upDiff=tuple([255 - colorThreshold] * 3),
                    flags=cv2.FLOODFILL_FIXED_RANGE)
                x, y, dx, dy = rect
                if dx > tileThreshold[0] and dy > tileThreshold[1]:
                    tile_img = self.screenImg[y:y + dy, x:x + dx, :]
                    # cv2.imshow('tile', tile_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    tileStr, res = self.classify(tile_img)
                    matched_tiles += [tileStr]
                    if len(matched_tiles) == 2:
                        print(matched_tiles)
                        if set(tiles) == set(matched_tiles):
                            print('\033[0;36mtiles matched: \033[0m: \033[0;32m{}\033[0m'.format(matched_tiles))
                            self.click(x, start[1])
                            return
                        matched_tiles = []
                    i = x + dx - start[0]
            else:
                fail += 1
            i += 1
