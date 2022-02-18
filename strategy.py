import functools
import random
import time

import numpy as np
from utils import *
from tile_masks import *
from Constants import *
import ctypes

DEBUG = False
DEBUG_DEFEND = True

TileValue = {
    'dora': 60,
    'dora_neighbour': 10,

    # 单张价值，作为其他类型组成成分时叠加计算价值
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 4,
    '7': 3,
    '8': 2,
    '9': 1,

    'selfwind': 10,
    'sanyuan': 10,
    'roundwind': 10,
}


class ScreenInfo:
    def __init__(self, RoundWind, SelfWind, doraTiles, handTiles,
                 myPaiHeTiles, xiaPaiHeTiles, duiPaiHeTiles, shangPaiHeTiles,
                 myFuLuTuiles, xiaFuLuTuiles, duiFuLuTuiles, shangFuLuTuiles,
                 yiPai):
        self.RoundWind = RoundWind
        self.SelfWind = SelfWind
        self.doraTiles = doraTiles
        self.handTiles = [tile[0] for tile in handTiles]

        self.myPaiHeTiles = myPaiHeTiles
        self.allPaiHeTiles_str = myPaiHeTiles, xiaPaiHeTiles, duiPaiHeTiles, shangPaiHeTiles
        self.myFuLuTiles = myFuLuTuiles
        self.allFuLuTuiles_str = myFuLuTuiles, xiaFuLuTuiles, duiFuLuTuiles, shangFuLuTuiles

        self.allPaiHeTiles = myPaiHeTiles[0] + myPaiHeTiles[1] \
                             + xiaPaiHeTiles[0] + xiaPaiHeTiles[1] \
                             + duiPaiHeTiles[0] + xiaPaiHeTiles[1] \
                             + shangPaiHeTiles[0] + xiaPaiHeTiles[1]
        self.allFuLuTiles = myFuLuTuiles + xiaFuLuTuiles + duiFuLuTuiles + shangFuLuTuiles
        self.yiPai = yiPai


class Strategy:
    def __init__(self, screenInfo: ScreenInfo):
        self.libc = ctypes.cdll.LoadLibrary("D:/github_workspace/codes/majsoul/action/cstrategy.so")
        self.screenInfo = screenInfo
        self.RoundWind = screenInfo.RoundWind
        self.SelfWind = screenInfo.SelfWind
        self.yiPai = screenInfo.yiPai
        self.doraStack = []

        self.menqing = len(screenInfo.myFuLuTiles) == 0  # 是否门清

        # 9*3+7=34
        self.handTiles = [0] * 34
        self.doraIndicator = [0] * 34
        self.allPaiHeTiles = [0] * 34
        self.allFuLuTiles = [0] * 34
        self.myFuLuTiles = [0] * 34
        self.myPaiHe = [0] * 34
        self.hiddenTiles = [4] * 34

        self.MianZi = 0
        self.DaZi = 0
        self.QueTou = 0

        self.DaZiStack = []

        self.maxValue = 0
        self.value = 0

        self.minShanTin = 8  # 普通型、七对
        # 最小向听时
        self.minMianZi = 0
        self.minDaZi = 0
        self.minQueTou = 0

        tile_transform_stack(screenInfo.doraTiles, self.doraStack)
        tile_transform_addbit2([(t, False) for t in screenInfo.handTiles] + screenInfo.myFuLuTiles,
                               self.handTiles)  # 手牌
        tile_transform_addbit1(screenInfo.doraTiles, self.doraIndicator)  # dora指示牌

        tile_transform_addbit2(screenInfo.allPaiHeTiles, self.allPaiHeTiles)  # 牌河
        tile_transform_addbit2(screenInfo.allFuLuTiles, self.allFuLuTiles)  # 副露
        tile_transform_addbit2(screenInfo.myFuLuTiles, self.myFuLuTiles)
        tile_transform_addbit2(screenInfo.myPaiHeTiles[0] + screenInfo.myPaiHeTiles[1], self.myPaiHe)
        self.calc_hidden_tiles()
        # print(self.handTiles)
        # print(self.myFuLuTiles)
        # print(self.doraIndicator)
        # print(self.myPaiHe)
        # print(self.hiddenTiles)

    def calc_hidden_tiles(self):
        for i in range(34):
            self.hiddenTiles[i] = 4 - (
                    self.handTiles[i] + self.allPaiHeTiles[i] + self.allFuLuTiles[i] - self.myFuLuTiles[i]
                    + self.doraIndicator[i])

    def get_minshantin(self, forced_guzhang_mask):
        handTile = tiles_encode(self.handTiles)
        fuluTile = tiles_encode(self.myFuLuTiles)
        forcedGuZhangTile = tiles_encode(forced_guzhang_mask)

        self.libc.get_shantin.restype = ctypes.c_char_p
        res = self.libc.get_shantin(handTile, fuluTile, forcedGuZhangTile).decode()
        res = [[ord(i) - ord('0') for i in r] for r in res.split('.')]

        res[1:] = [[id2str[i] for i in j] for j in res[1:]]
        return res[0][0], res[1:]  # (shantin数，拆分)

    def get_maxvalue(self, forced_guzhang_mask):
        handTile = tiles_encode(self.handTiles)
        fuluTile = tiles_encode(self.myFuLuTiles)
        forcedGuZhangTile = tiles_encode(forced_guzhang_mask)
        return self.libc.get_value(handTile, fuluTile, forcedGuZhangTile)

    def get_JinZhang(self, lastShantin, forced_guzhang_mask):
        tmp = [0] * 34
        # 确定能使向听数减少的进张，34次深度优先搜索
        for i in range(34):
            self.handTiles[i] += 1
            new_minshantin, chaifen = self.get_minshantin(forced_guzhang_mask)
            if new_minshantin < lastShantin:
                tmp[i] = 1
            self.handTiles[i] -= 1
        N = 0
        for k in range(34):
            if tmp[k] == 1:
                if k <= 26:
                    if id2str[k] in self.doraStack:  # dora更可能被别人捏在手里
                        N += self.hiddenTiles[k] // 2
                    else:
                        N += self.hiddenTiles[k]
                else:  # 字牌
                    if self.hiddenTiles[k] == 2:  # 剩余2张未出现更可能有人手中有一对
                        N += self.hiddenTiles[k] // 2
                    else:
                        N += self.hiddenTiles[k]
        jinzhangTileNames = {id2str[k]: self.hiddenTiles[k] for k in range(34) if tmp[k] == 1}
        return N, jinzhangTileNames

    def isZhenTing(self, names):
        for key in names:
            if key in [info[0] for info in self.screenInfo.myPaiHeTiles[0]]:
                return True
        return False

    # 听牌时决策，只考虑第一役种
    def actionTingPai(self, records):
        if len(records) == 0:
            return []
        # 好型n(多面听) > 好型2 > 双碰 > 坎张 > 单骑
        res = []
        for record in records:
            tile, yis, op = record
            (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
            if len(names) >= 3:
                res.append((record, HAOXINGn, -1))
            elif len(names) == 2:  # 听2张
                tmp = []
                for c in chaifen:
                    if len(c) == 2:
                        tmp.append(c)
                if len(tmp) == 2:  # 双碰
                    res.append((record, SHUANGPENG, -1))
                else:
                    res.append((record, HAOXING2, -1))
            elif len(names) == 1 and jinzhang != 0:  # 听1张
                t, n = list(enumerate(names.items()))[0][1]
                d = dist_to_side(t) if n > 0 else 100
                for c in chaifen:
                    if len(c) == 1:  # 单骑
                        res.append((record, DANJI, d))
                    elif len(c) == 2:  # 坎张
                        res.append((record, KANZHANG, d))
        res.sort(key=lambda record: record[1], reverse=False)  # 按优先级升序
        if res[0][1] >= KANZHANG:  # 只听一种牌
            res.sort(key=lambda record: record[2], reverse=False)  # 按进张到边张的距离升序
        res = [r[0] for r in res]
        if DEBUG:
            print('after tingpai:', res)
        return res

    def Yi(self, ignore_menqing=False, threshold=4):  # 副露时判断是否有役，若为0则表明有役，为正值则为与有役的最小距离
        dists = {}
        if not ignore_menqing and self.menqing:
            dists['menqing'] = 0, MASK_FLAG
        # 自/场风与三元牌役
        tmp = [str2id[self.screenInfo.SelfWind], str2id[self.screenInfo.RoundWind], 31, 32, 33]
        for idx in tmp:
            if self.handTiles[idx] >= 3:
                dists[id2str[idx]] = 0, MASK_FLAG
            elif self.handTiles[idx] + self.hiddenTiles[idx] >= 3:
                dists[id2str[idx]] = 2 * (3 - self.handTiles[idx]), MASK_FLAG  # 役牌的距离翻倍
            else:
                dists[id2str[idx]] = 100, MASK_FLAG  # 距离无穷大
        # 对对和，即手牌种类加起来共5种且全没有顺子副露，而任何手牌种类数必然不小于5，以len(duidui) - 5为距离
        # 不考虑副露有同色三顺子情形
        duidui = []
        duidui_p = True
        for i in range(34):
            if 1 <= self.myFuLuTiles[i] <= 2:
                duidui_p = False
        if duidui_p:
            for i in range(34):
                if self.handTiles[i] > 0:
                    duidui.append(i)
            dists['duidui'] = len(duidui) - 5, MASK_FLAG
        # 大三元
        sanyuan_d = 0
        for i in range(31, 34):
            if self.handTiles[i] + self.hiddenTiles[i] >= 3:
                sanyuan_d += 2 - self.handTiles[i] if self.handTiles[i] <= 2 else 0
            else:
                sanyuan_d += 100
        dists['dasanyuan'] = sanyuan_d, MASK_FLAG
        # 特殊役种1，mask位为0则手牌对应位必须是0
        for key in YiZhongMasks:
            this_min_dist = 100
            this_min_mask = None
            for mask in YiZhongMasks[key]:
                dist = 0
                for i in range(34):
                    if mask[i] == 0 and self.handTiles[i] != 0:
                        dist += self.handTiles[i] + 100 * self.myFuLuTiles[i]
                if dist < this_min_dist:
                    this_min_dist = dist
                    this_min_mask = mask
            # 调权值
            # 混一色距离减小
            if key == 'hunyise':
                if this_min_dist <= 5:  # 其余色<=5张就把混一色加入考虑
                    dists[key] = max(0, this_min_dist / 2), this_min_mask
            else:
                dists[key] = this_min_dist, this_min_mask

        # 特殊役种2，若mask位为1则手牌对应位必须是1
        for key in YiZhongMasks2:
            this_min_dist = 100
            this_min_mask = None
            for mask in YiZhongMasks2[key]:
                dist = 0
                for i in range(34):
                    if mask[i] == 1 and self.handTiles[i] == 0:
                        dist += 2
                if dist < this_min_dist:
                    this_min_dist = dist
                    this_min_mask = mask
            # 调权值
            dists[key] = this_min_dist, this_min_mask
        # 按dist升序排序
        dists = sorted(dists.items(), key=lambda x: x[1][0], reverse=False)
        dists = [dist for dist in dists if dist[1][0] <= threshold]  # 只保留距离小于等于threshold的役
        return dists

    def is_guzhang(self, k):
        if k >= 27:
            return self.handTiles[k] <= 1
        res = False
        for i in range(-2, 3):
            if i != 0 and 0 <= k % 9 + i <= 8:
                res = res or self.handTiles[k + i] > 0
        return not res and self.handTiles[k] == 1

    def is_dora(self, k):
        return id2str[k] in self.doraStack

    def calc_alldoras_value(self, discardTile=None):  # dora还给予了其相邻数牌额外价值
        res = 0
        doras = [str2id[name] for name in self.doraStack]
        for dname, name in [('0m', '5m'), ('0p', '5p'), ('0s', '5s')]:
            if dname in self.screenInfo.handTiles:  # 手牌中有红dora，如果弃的牌是唯一的5m，则不计红5m的价值
                if discardTile is not None and discardTile == name and self.handTiles[str2id[name]] != 0:
                    # 此时handTiles已经减1
                    res += TileValue['dora']

        for i in doras:
            if self.handTiles[i] > 0:
                res += self.handTiles[i] * TileValue['dora']
                if i <= 26:
                    k = i % 9
                    if k >= 1:
                        res += self.handTiles[i - 1] * TileValue['dora_neighbour']
                    if k >= 2:
                        res += self.handTiles[i - 2] * TileValue['dora_neighbour']
                    if k <= 7:
                        res += self.handTiles[i + 1] * TileValue['dora_neighbour']
                    if k <= 6:
                        res += self.handTiles[i + 2] * TileValue['dora_neighbour']
        return res

    def calc_alltiles_value(self):  # 计算所有手牌单张价值和
        res = 0
        for i in range(34):
            if self.handTiles[i] > 0:
                if i <= 26:
                    res += self.handTiles[i] * TileValue[id2str[i][0]]
                elif self.hiddenTiles[i] + self.handTiles[i] >= 3 and self.handTiles[i] + self.myFuLuTiles[i] <= 2:
                    if len(self.screenInfo.myPaiHeTiles[0] + self.screenInfo.myPaiHeTiles[1]) <= 8 \
                            or self.handTiles[i] >= 2:
                        val = 0
                        if 31 <= i <= 33:  # 场上出现张数越多，巡数越多价值越低，2张价值减为10
                            val = TileValue['sanyuan'] - 45 * self.allPaiHeTiles[i]
                        if str2id[self.screenInfo.SelfWind] == i:
                            val = TileValue['selfwind'] - 45 * self.allPaiHeTiles[i]
                        if str2id[self.screenInfo.RoundWind] == i:
                            val = TileValue['roundwind'] - 45 * self.allPaiHeTiles[i]
                        res += self.handTiles[i] * val
        return res

    # 防守
    def defend(self):
        danger_levels = [3] * 4
        # 根据副露判断危险牌，根据立直与否判断安全牌
        danger_masks = [[[0] * 34, False], [[0] * 34, False], [[0] * 34, False], [[0] * 34, False], ]  # 为0不能打
        safe_masks = [[[0] * 34, False], [[0] * 34, False], [[0] * 34, False], [[0] * 34, False], ]  # 为1绝对安全

        for d in range(1, 4):
            if len(self.screenInfo.allPaiHeTiles_str[d][1]) > 0:  # 有横置的牌，立直
                danger_levels[d] = DANGER_LEVEL_2
                continue
            # 有副露
            # 清一色副露牌数>=6，字一色副露数>=9
            if len(self.screenInfo.allFuLuTuiles_str[d]) >= 6:
                flags = [True, True, True]  # m/p/s
                for tile, rotated in self.screenInfo.allFuLuTuiles_str[d]:
                    if tile[1] != 'm':
                        flags[0] = False
                    if tile[1] != 'p':
                        flags[1] = False
                    if tile[1] != 's':
                        flags[2] = False
                if flags[0] or flags[1] or flags[2]:  # 清一色
                    if DEBUG_DEFEND:
                        print('fulu qingyise:', flags)
                        print(self.screenInfo.allFuLuTuiles_str)
                    danger_levels[d] = DANGER_LEVEL_1
                for dd in range(0, 3):
                    if flags[dd]:
                        danger_masks[d] = tMask[dd].copy(), True
            if len(self.screenInfo.allFuLuTuiles_str[d]) >= 9:
                flag = True  # m/p/s
                for tile, rotated in self.screenInfo.allFuLuTuiles_str[d]:
                    if tile[1] != 'z':
                        flag = False
                if flag:  # 字一色
                    if DEBUG_DEFEND:
                        print('fulu ziyise:')
                        print(self.screenInfo.allFuLuTuiles_str)
                    danger_levels[d] = DANGER_LEVEL_1
                danger_masks[d] = zMask.copy(), True

            # dora >= 3
            dora_n = 0
            for tile, rotated in self.screenInfo.allFuLuTuiles_str[d]:
                if tile in self.doraStack:
                    dora_n += 1
            if dora_n >= 3:
                if DEBUG_DEFEND:
                    print('fulu doras:', d, dora_n)
                    print(self.screenInfo.allFuLuTuiles_str)
                danger_levels[d] = DANGER_LEVEL_1
            elif dora_n == 2:
                if DEBUG_DEFEND:
                    print('fulu doras:', d, dora_n)
                    print(self.screenInfo.allFuLuTuiles_str)
                danger_levels[d] = DANGER_LEVEL_3

            # 混一色
            if len(self.screenInfo.allFuLuTuiles_str[d]) >= 9:
                flags = [True, True, True, False]  # m/p/s
                for tile, rotated in self.screenInfo.allFuLuTuiles_str[d]:
                    if tile[1] != 'z':
                        if tile[1] != 'm':
                            flags[0] = False
                        if tile[1] != 'p':
                            flags[1] = False
                        if tile[1] != 's':
                            flags[2] = False
                    else:
                        flags[3] = True
                if flags[3] and (flags[0] or flags[1] or flags[2]):  # 混一色
                    if DEBUG_DEFEND:
                        print('fulu hunyise:', flags)
                        print(self.screenInfo.allFuLuTuiles_str)
                    danger_levels[d] = DANGER_LEVEL_3
                for dd in range(0, 3):
                    if flags[dd]:
                        danger_masks[d] = tzMask[dd].copy(), True

        # 针对立直的防守
        liqi = [(False, -1)] * 4

        if len(self.screenInfo.allPaiHeTiles_str[1][1]) > 0:
            liqi[1] = True, len(self.screenInfo.allPaiHeTiles_str[1][0])
        if len(self.screenInfo.allPaiHeTiles_str[2][1]) > 0:
            liqi[2] = True, len(self.screenInfo.allPaiHeTiles_str[2][0])
        if len(self.screenInfo.allPaiHeTiles_str[3][1]) > 0:
            liqi[3] = True, len(self.screenInfo.allPaiHeTiles_str[3][0])

        for d in range(1, 4):
            if liqi[d][0]:
                discardTileInXiajiaFulu_cnt = 0  # 被下家吃的牌数
                # 舍张振听
                for tile, rotated in self.screenInfo.allPaiHeTiles_str[d][0] + self.screenInfo.allPaiHeTiles_str[d][1]:
                    safe_masks[d][1] = True
                    safe_masks[d][0][str2id[tile]] = SAFE_LEVEL_1
                    # 筋牌
                    if tile[0] in ['4', '5', '6', '0'] and tile[1] in ['m', 'p', 's']:
                        tile_0 = tile[0] if tile[0] != '0' else '5'
                        safe_masks[d][1] = True
                        safe_masks[d][0][str2id['{}{}'.format(chr(ord(tile_0) - 3), tile[1])]] = SAFE_LEVEL_3
                        safe_masks[d][0][str2id['{}{}'.format(chr(ord(tile_0) + 3), tile[1])]] = SAFE_LEVEL_3
                # 立直方下家副露的横置牌，且其所在的区间内是顺子，即吃牌（碰也有可能，但不考虑）
                k = 0
                for tile, rotated in self.screenInfo.allFuLuTuiles_str[(d + 1) % 4]:
                    if rotated:
                        # 确认区间[k // 3 * 3, k // 3 * 3 + 2]内牌是顺子
                        int_tiles = []
                        for p in range(k // 3 * 3, k // 3 * 3 + 3):
                            int_tiles.append(self.screenInfo.allFuLuTuiles_str[(d + 1) % 4][p][0])
                        flag = is_shunzi(int_tiles)
                        if flag:
                            if DEBUG_DEFEND:
                                print('liqi xiajia chipai:', tile)
                                print(self.screenInfo.allFuLuTuiles_str)
                            discardTileInXiajiaFulu_cnt += 1
                            safe_masks[d][1] = True
                            safe_masks[d][0][str2id[tile]] = SAFE_LEVEL_1
                            # 筋牌
                            if tile[0] in ['4', '5', '6'] and tile[1] in ['m', 'p', 's']:
                                safe_masks[d][1] = True
                                safe_masks[d][0][
                                    str2id['{}{}'.format(chr(ord(tile[0]) - 3), tile[1])]] = SAFE_LEVEL_3
                                safe_masks[d][0][
                                    str2id['{}{}'.format(chr(ord(tile[0]) + 3), tile[1])]] = SAFE_LEVEL_3
                    k += 1
                # 立直振听
                for d2 in range(4):
                    if d2 != d:
                        tmp = self.screenInfo.allPaiHeTiles_str[d2][0] + self.screenInfo.allPaiHeTiles_str[d2][1]
                        tmp = tmp[liqi[d][1] + discardTileInXiajiaFulu_cnt:]
                        for tile, rotated in tmp:
                            safe_masks[d][1] = True
                            safe_masks[d][0][str2id[tile]] = SAFE_LEVEL_1

            # 字牌
            for i in range(27, 34):
                if safe_masks[d][0][i] != SAFE_LEVEL_1:
                    safe_masks[d][0][i] = SAFE_LEVEL_2

        # 至此得到了所有方的安全牌和危险牌、立直信息，综合决策
        syn_safe_tiles_0 = [True] * 34  # 绝对安全，所有方安全牌的交集
        valid_0 = False
        syn_safe_tiles_1 = [False] * 34  # 至少对一家安全，所有方安全牌的并集
        syn_safe_tiles_2 = [False] * 34  # 可能安全牌
        for d in range(1, 4):
            if safe_masks[d][1]:
                valid_0 = True
            for i in range(34):
                if safe_masks[d][1] and safe_masks[d][0][i] != SAFE_LEVEL_1:
                    syn_safe_tiles_0[i] = False
                if safe_masks[d][1] and safe_masks[d][0][i] == SAFE_LEVEL_1:
                    syn_safe_tiles_1[i] = True
                if safe_masks[d][1] and safe_masks[d][0][i] != 0:
                    syn_safe_tiles_2[i] = True

        if not valid_0:
            for i in range(34):
                syn_safe_tiles_0[i] = False

        syn_safe_tiles = (syn_safe_tiles_0, syn_safe_tiles_1, syn_safe_tiles_2)
        syn_danger_tiles = [-1] * 34
        exist_dangerous_fulu = False
        # 取最大危险度
        for i in range(34):
            for d in range(1, 4):
                if danger_masks[d][1] and danger_masks[d][0][i]:
                    exist_dangerous_fulu = True
                if danger_masks[d][1] and danger_masks[d][0][i] > syn_danger_tiles[i]:
                    syn_danger_tiles[i] = danger_masks[d][0][i]

        exist_liqi = liqi[1][0] or liqi[2][0] or liqi[3][0]
        defense_applied = exist_liqi or exist_dangerous_fulu

        return defense_applied, syn_safe_tiles, syn_danger_tiles, [l[0] for l in liqi]

    def getNextAction(self, lastDiscardTile=None, ops=None):
        if ops is None:
            ops = [Operation.Ordinary]
        if len(set(self.screenInfo.handTiles).union(set(self.screenInfo.myFuLuTiles))) < 5:  # 输入有误
            print('error,', self.screenInfo.handTiles + self.myFuLuTiles)
            return None, None, None, None
        # 进攻
        # 优先级：最小向听数 > 最多进张数 > 最大价值
        # 首先满足最小向听数，然后在进张数的前3名中选择价值最大的

        records = []  # 记录所有选择

        for op in ops:
            if op == Operation.NoEffect:
                # forced_guzhang_mask：若mask某位为0，则手牌对应位必须拆解为孤张
                yis = self.Yi(ignore_menqing=not self.menqing)
                yi_infos = []
                for yi in yis:
                    name, (dist, forced_guzhang_mask) = yi
                    min_shantin, chaifen = self.get_minshantin(forced_guzhang_mask)

                    single_tile_values = self.calc_alldoras_value() + self.calc_alltiles_value()
                    maxValue = single_tile_values + self.get_maxvalue(forced_guzhang_mask)
                    max_jinzhang, names = self.get_JinZhang(min_shantin, forced_guzhang_mask)

                    yipai_in_jinzhang = False
                    for tile in names:
                        if tile in self.yiPai:
                            yipai_in_jinzhang = True
                            break
                    # print(idx2str[i], self.myPaiHe, names, self.isZhenTing(names))
                    # 不振听听牌, 不无役听牌，但允许听役牌
                    if min_shantin >= 1 or (not self.isZhenTing(names) and (dist == 0 or yipai_in_jinzhang)):
                        # 去掉役种不允许的进张
                        for p in range(34):
                            if forced_guzhang_mask[p] == 0 and id2str[p] in names:
                                max_jinzhang -= names[id2str[p]]
                                del names[id2str[p]]
                        yi_infos.append(((name, dist), min_shantin, max_jinzhang,
                                         maxValue, names, chaifen, set()))
                yi_infos = sorted(yi_infos, key=lambda x: x[0][1], reverse=False)  # 按距离升序
                if len(yi_infos) != 0:
                    records.append((None, yi_infos, op))

                continue
            # 碰：直接副露加入3张；吃：当做普通摸牌处理，然后从拆分中找出一个含有所吃的牌的顺子即可
            elif op == Operation.Chi:
                tile_transform_addbit1([lastDiscardTile], self.handTiles)
            elif op == Operation.Peng:
                tile_transform_addbit1([lastDiscardTile], self.handTiles)
                if self.handTiles[str2id[lastDiscardTile]] < 3:
                    tile_transform_subbit1([lastDiscardTile], self.handTiles)
                    continue
                tile_transform_addbit1([lastDiscardTile], self.myFuLuTiles)
                tile_transform_addbit1([lastDiscardTile], self.myFuLuTiles)
                tile_transform_addbit1([lastDiscardTile], self.myFuLuTiles)

            for i in range(34):
                if self.handTiles[i] - self.myFuLuTiles[i] > 0:  # 不打已副露的牌
                    self.handTiles[i] -= 1

                    # forced_guzhang_mask：若mask某位为0，则手牌对应位必须拆解为孤张
                    yis = self.Yi(ignore_menqing=not self.menqing or lastDiscardTile is not None)
                    yi_infos = []
                    for yi in yis:
                        if op in [Operation.Chi, Operation.Peng] and yi[0] == 'menqing':
                            continue
                        name, (dist, forced_guzhang_mask) = yi
                        min_shantin, chaifen = self.get_minshantin(forced_guzhang_mask)

                        single_tile_values = self.calc_alldoras_value(id2str[i]) + self.calc_alltiles_value()
                        maxValue = single_tile_values + self.get_maxvalue(forced_guzhang_mask)
                        max_jinzhang, names = self.get_JinZhang(min_shantin, forced_guzhang_mask)

                        yipai_in_jinzhang = False
                        for tile in names:
                            if tile in self.yiPai:
                                yipai_in_jinzhang = True
                                break
                        # print(idx2str[i], self.myPaiHe, names, self.isZhenTing(names))
                        # 不振听听牌, 不无役听牌，但允许听役牌
                        if min_shantin >= 1 or (not self.isZhenTing(names) and (dist == 0 or yipai_in_jinzhang)):
                            # 去掉役种不允许的进张
                            for p in range(34):
                                if forced_guzhang_mask[p] == 0 and id2str[p] in names:
                                    max_jinzhang -= names[id2str[p]]
                                    del names[id2str[p]]
                            # 吃/碰
                            if op == Operation.Chi:
                                for c in chaifen:
                                    flag = True  # c不能是副露
                                    for p in c:
                                        if self.handTiles[str2id[p]] - self.myFuLuTiles[str2id[p]] == 0:
                                            flag = False

                                    if flag and lastDiscardTile in c and len(c) == 3:
                                        choice = set(c) - {lastDiscardTile}
                                        if is_shunzi(c):  # 找到含有lastDiscardTile的顺子
                                            tile_transform_addbit1(c, self.myFuLuTiles)  # 加入副露

                                            new_yis = self.Yi(ignore_menqing=True)
                                            if len(new_yis) > 0 and new_yis[0][1][0] <= 2:
                                                # 吃碰后最小役距离<=2，若len为0则表示吃碰后无距离<=4的役
                                                min_shantin, chaifen = self.get_minshantin(forced_guzhang_mask)
                                                single_tile_values = self.calc_alldoras_value(
                                                    id2str[i]) + self.calc_alltiles_value()
                                                maxValue = single_tile_values + self.get_maxvalue(forced_guzhang_mask)
                                                max_jinzhang, names = self.get_JinZhang(min_shantin,
                                                                                        forced_guzhang_mask)

                                                tile_transform_subbit1(c, self.myFuLuTiles)  # 还原

                                                yi_infos.append(((name, dist), min_shantin, max_jinzhang,
                                                                 maxValue, names, chaifen, choice))
                                                break  # 找到一个即可
                            else:
                                yi_infos.append(((name, dist), min_shantin, max_jinzhang,
                                                 maxValue, names, chaifen, set()))

                    yi_infos = sorted(yi_infos, key=lambda x: x[0][1], reverse=False)  # 按距离升序
                    if len(yi_infos) != 0:
                        records.append((id2str[i], yi_infos, op))
                        # print(idx2str[i])
                        # self.print_records([records[-1]])

                    self.handTiles[i] += 1

            if op == Operation.Chi:
                tile_transform_subbit1([lastDiscardTile], self.handTiles)
            elif op == Operation.Peng:
                tile_transform_subbit1([lastDiscardTile], self.handTiles)
                tile_transform_subbit1([lastDiscardTile], self.myFuLuTiles)
                tile_transform_subbit1([lastDiscardTile], self.myFuLuTiles)
                tile_transform_subbit1([lastDiscardTile], self.myFuLuTiles)

        min_dist = 100
        min_shantin = 8

        if DEBUG:
            print('flag1')
            print_records(records)

        # 第一役种最小向听预判断，先判断是否防守，决策放在后面
        for record in records:
            tile, yis, op = record
            (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
            if shantin < min_shantin:
                min_shantin = shantin

        # 防守判断，若有人立直并且自己向听数>=2，考虑弃和；否则跳过
        # 只打以下牌：立直方的牌河牌、立直巡数后的所有牌河牌、部分筋牌
        defense_applied, syn_safe_tiles, syn_danger_tiles, liqi_condition = self.defend()
        absolutely_safe_tiles = []
        specific_safe_tiles = []
        maybe_safe_tiles = []

        for i in range(34):
            if syn_safe_tiles[0][i]:
                absolutely_safe_tiles.append(id2str[i])
            if syn_safe_tiles[1][i]:
                specific_safe_tiles.append(id2str[i])
            if syn_safe_tiles[2][i]:
                maybe_safe_tiles.append(id2str[i])

        # print(absolutely_safe_tiles)
        # print(specific_safe_tiles)
        # print(maybe_safe_tiles)
        if not defense_applied:
            print('\033[0;32mno defense\033[0m')
        else:
            if min_shantin <= 0:
                # 只剔除危险等级最高的打法
                records = [record for record in records
                           if record[0] is None or syn_danger_tiles[str2id[record[0]]] != DANGER_LEVEL_1]
                print('tingpai, only defend most dangerous')
            else:
                print('defense applied')
                tmp = [record for record in records if record[0] in absolutely_safe_tiles]
                tmp2 = [record for record in records if record[0] in specific_safe_tiles]
                tmp3 = [record for record in records if record[0] in maybe_safe_tiles]
                if len(tmp) > 0:  # 有安全牌的打法，仍有进攻机会
                    print('defend with attack')
                    print_tiles('absolutely_safe_tiles', absolutely_safe_tiles)
                    records = tmp
                elif len(tmp2) > 0:
                    print('partial defend with attack')
                    print_tiles('specific_safe_tiles', specific_safe_tiles)
                    records = tmp2
                elif len(tmp3) > 0:
                    print('prob_defend with attack')
                    print_tiles('maybe_safe_tiles', maybe_safe_tiles)
                    records = tmp3
                else:  # 完全弃和，强制打出一张安全牌
                    defend_tile = None
                    for i in range(34):
                        if id2str[i] in syn_safe_tiles and self.handTiles[i] - self.myFuLuTiles[i] > 0:
                            defend_tile = id2str[i]
                            break
                    if defend_tile is not None:
                        print('\033[0;31mdefend: {}, shantin={}\033[0m'.format(defend_tile, min_shantin))
                        if lastDiscardTile is not None:
                            tile_transform_subbit1([lastDiscardTile], self.handTiles)
                        return defend_tile, None, False
                    else:  # 手牌没有安全牌，放弃防守
                        print('no safe tiles, give up defense')

        safe_tiles = ((absolutely_safe_tiles, 'absolutely_safe'),
                      (specific_safe_tiles, 'specific_safe_tiles'),
                      (maybe_safe_tiles, 'maybe_safe_tiles'),)
        if DEBUG:
            print('flag2')
            print_records(records, safe_tiles)
        # 选择NoEffect以及吃碰后第一距离最小打法，并且第一距离最小必须<=2
        for record in records:
            tile, yis, op = record
            if op != Operation.NoEffect and len(yis) > 0:
                (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
                if dist < min_dist:
                    min_dist = dist
        records = [record for record in records if record[2] == Operation.NoEffect
                   or (record[1][0][0][1] == min_dist and (min_dist <= 2 or defense_applied))]
        if DEBUG:
            print('flag3')
            print_records(records, safe_tiles)

        min_shantin = 8
        # 再次判断第一役种最小向听
        for record in records:
            tile, yis, op = record
            (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
            if shantin < min_shantin:
                min_shantin = shantin
        records = [record for record in records if
                   record[1][0][1] == min_shantin and (record[1][0][2] > 0 or defense_applied)]

        if DEBUG:
            print('flag4')
            print_records(records, safe_tiles)

        if min_shantin == 0:
            records = self.actionTingPai(records)

        if DEBUG:
            print('flag5')
            print_records(records, safe_tiles)

        records.sort(key=lambda record: record[1][0][3], reverse=True)  # 按价值降序
        if len(records) >= 6:  # 第一役种价值取前1/2
            records = records[:len(records) // 2 + 1]

        if DEBUG:
            print('flag6')
            print_records(records, safe_tiles)

        # 按第二距离升序，没有第二距离则按第二距离为1参与排序，以给第二距离为0的record更高优先级
        records.sort(key=lambda record: record[1][1][0][1] if len(record[1]) >= 2 else 1)
        if len(records) > 0 and records[0][1][0][1] <= 2:
            records = records[:5]

        if DEBUG:
            print('flag7')
            print_records(records, safe_tiles)

        # 第一役种最多进张
        records.sort(key=lambda record: record[1][0][2], reverse=True)  # 按进张数降序排序
        records = records[:5]

        if DEBUG:
            print('flag8')
            print_records(records, safe_tiles)

        # 价值前4名
        records.sort(key=lambda record: record[1][0][3], reverse=True)  # 按价值降序
        syn_res = records[:4]
        # print('syn_res:', syn_res)
        if DEBUG:
            print('flag9')
            print_records(syn_res, safe_tiles)

        if len(syn_res) == 0:
            if lastDiscardTile is not None:
                tile_transform_subbit1([lastDiscardTile], self.handTiles)
            return None, None, None, None

        syn_choice = syn_res[0]

        if lastDiscardTile is not None:
            tile, yis, op = syn_choice
            (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
            if op == Operation.NoEffect or is_shunzi(choice.union({tile})):
                # 吃牌后要弃的牌不能与另外两张牌组成顺子
                # print(tile, choice, op2str[op])
                tile_transform_subbit1([lastDiscardTile], self.handTiles)
                return None, None, None, None  # (tile, choice, liqi)
            print('chi/peng:', lastDiscardTile, tile, choice, op2str[op])
            print_records([syn_choice], safe_tiles)
            return tile, op, choice, False  # (tile, op, choice, liqi)

        print_records([syn_choice], safe_tiles)
        return syn_choice[0], syn_choice[2], syn_choice[1][0][1], self.menqing and syn_choice[1][0][
            1] == 0  # (tile, shantin, liqi)
