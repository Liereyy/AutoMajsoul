import functools
import random
import time

import numpy as np
from utils import *
import ctypes

mask_base = [i for i in range(34)]
MASK_FLAG = [1] * 34  # 所有手牌都可以存在

DEBUG = False

# 判断是否有役，mask为2层列表
# 若mask位为0则手牌对应位必须是0
YiZhongMasks = {
    'duanyaojiu': [[1 if i <= 26 and 1 <= i % 9 <= 7 else 0 for i in mask_base]],
    'hunyise': [[1 if (i <= 26 and 9 * k <= i <= 9 * k + 8) or i >= 27 else 0 for i in mask_base] for k in range(3)],
    'hunlaotou': [[1 if (i <= 26 and (i == 9 * k or i == 9 * k + 8)) or i >= 27 else 0 for i in mask_base] for k in
                  range(3)],
}

# 若mask位为1则手牌对应位必须是1
YiZhongMasks2 = {
    'sansetongshun': [[1 if (i <= 26 and (p <= i <= p + 2 or p + 9 <= i <= p + 9 + 2 or p + 18 <= i <= p + 18 + 2))
                       else 0 for i in mask_base] for p in range(7)],
}

TileValue = {
    'dora': 600,
    'dora_neighbour': 100,

    # 单张价值，作为其他类型组成成分时叠加计算价值
    '1': 10,
    '2': 20,
    '3': 30,
    '4': 40,
    '5': 50,
    '6': 40,
    '7': 30,
    '8': 20,
    '9': 10,

    'selfwind': 100,
    'sanyuan': 100,
    'roundwind': 100,
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

        self.minShanTin = 8  # 不考虑七对和国士
        # 最小向听时
        self.minMianZi = 0
        self.minDaZi = 0
        self.minQueTou = 0

        tile_transform_dora(screenInfo.doraTiles, self.doraStack)
        tile_transform_addbit(screenInfo.handTiles + screenInfo.myFuLuTiles, self.handTiles)  # 手牌
        tile_transform_addbit(screenInfo.doraTiles, self.doraIndicator)  # dora指示牌
        tile_transform_addbit(screenInfo.allPaiHeTiles, self.allPaiHeTiles)  # 牌河
        tile_transform_addbit(screenInfo.allFuLuTiles, self.allFuLuTiles)  # 副露
        tile_transform_addbit(screenInfo.myFuLuTiles, self.myFuLuTiles)
        tile_transform_addbit(screenInfo.myPaiHeTiles, self.myPaiHe)
        self.calc_hidden_tiles()
        # print(self.handTiles)
        # print(self.fuLuTiles)
        # print(self.doraIndicator)
        # print(self.paiHeTiles)
        # print(self.hiddenTiles)

    def calc_hidden_tiles(self):
        for i in range(34):
            self.hiddenTiles[i] = 4 - (
                    self.handTiles[i] + self.allPaiHeTiles[i] + self.allFuLuTiles[i] + self.doraIndicator[i]
                    - self.myFuLuTiles[i])

    def get_minshantin(self, forced_guzhang_mask):
        handTileCode = ''
        for k in self.handTiles:
            handTileCode += str(k)
        handTile = handTileCode.encode()

        fuLuCode = ''
        for k in self.myFuLuTiles:
            fuLuCode += str(k)
        fuluTile = fuLuCode.encode()

        forcedGuZhangCode = ''
        for k in forced_guzhang_mask:
            forcedGuZhangCode += str(k)
        forcedGuZhangTile = forcedGuZhangCode.encode()

        self.libc.get_shantin.restype = ctypes.c_char_p
        res = self.libc.get_shantin(handTile, fuluTile, forcedGuZhangTile).decode()
        res = [[ord(i) - ord('0') for i in r] for r in res.split('.')]

        res[1:] = [[idx2str[i] for i in j] for j in res[1:]]
        return res[0][0], res[1:]  # shantin数

    def get_maxvalue(self, forced_guzhang_mask):
        handTileCode = ''
        for k in self.handTiles:
            handTileCode += str(k)
        handTile = handTileCode.encode()
        fuLuCode = ''
        for k in self.myFuLuTiles:
            fuLuCode += str(k)
        fuluTile = fuLuCode.encode()
        forcedGuZhangCode = ''
        for k in forced_guzhang_mask:
            forcedGuZhangCode += str(k)
        forcedGuZhangTile = forcedGuZhangCode.encode()
        return self.libc.get_value(handTile, fuluTile, forcedGuZhangTile)

    def get_JinZhang(self, lastShantin):
        tmp = [0] * 34
        # 确定能使向听数减少的进张，34次深度优先搜索
        for i in range(34):
            self.handTiles[i] += 1
            cur_minshantin, chaifen = self.get_minshantin(MASK_FLAG)
            if cur_minshantin < lastShantin:
                tmp[i] = 1
            self.handTiles[i] -= 1
        N = 0
        for k in range(34):
            if tmp[k] == 1:
                N += self.hiddenTiles[k]
        jinzhangTileNames = {idx2str[k]: self.hiddenTiles[k] for k in range(34) if tmp[k] == 1}
        return N, jinzhangTileNames

    def isZhenTing(self, names):
        for key in names:
            if key in self.screenInfo.myPaiHeTiles[0]:
                return True
        return False

    # 听牌时决策，只考虑第一役种
    def actionTingPai(self, records):
        # 优先听多种牌，只听一种牌时优先边张
        res = []
        one_choice = []
        for record in records:
            tile, yis = record
            (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
            if len(names) >= 2:
                res.append(record)
            elif len(names) == 1 and jinzhang != 0:
                tile, n = list(enumerate(names.items()))[0]
                one_choice.append((record, dist_to_side(idx2str[tile])))
        if len(one_choice) != 0:
            one_choice.sort(key=lambda record: record[1], reverse=True)  # 按到边张距离降序
            res.append(one_choice[0][0])
        if DEBUG:
            print('one choice:', one_choice)
        return res

    def Yi(self, ignore_menqing=False):  # 副露时判断是否有役，若为0则表明有役，为正值则为与有役的最小距离
        dists = {}
        if not ignore_menqing and self.menqing:
            dists['menqing'] = 0, MASK_FLAG
        # 自/场风与三元牌役
        tmp = [str2id[self.screenInfo.SelfWind], str2id[self.screenInfo.RoundWind], 31, 32, 33]
        for idx in tmp:
            if self.handTiles[idx] >= 3:
                dists[idx2str[idx]] = 0, MASK_FLAG
            elif self.handTiles[idx] + self.hiddenTiles[idx] >= 3:
                dists[idx2str[idx]] = 2 * (3 - self.handTiles[idx]), MASK_FLAG  # 役牌的距离翻倍
            else:
                dists[idx2str[idx]] = 100, MASK_FLAG  # 距离无穷大
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
        # 特殊役种1
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

        # 特殊役种2
        for key in YiZhongMasks2:
            this_min_dist = 100
            this_min_mask = None
            for mask in YiZhongMasks2[key]:
                dist = 0
                for i in range(34):
                    if mask[i] == 1 and self.handTiles[i] == 0:
                        dist += 2
                    if mask[i] == 0 and self.myFuLuTiles[i] != 0:
                        dist += 100 * self.myFuLuTiles[i]
                if dist < this_min_dist:
                    this_min_dist = dist
                    this_min_mask = mask
            # 调权值
            dists[key] = this_min_dist, this_min_mask
        # 按dist升序排序
        dists = sorted(dists.items(), key=lambda x: x[1][0], reverse=False)
        dists = [dist for dist in dists if dist[1][0] <= 4]  # 只保留距离小于等于4的役
        return dists

    def get_fan(self, chaifen):  # 计算番数
        fan = 0
        if len(self.myFuLuTiles) == 0:  # 门清立直
            fan += 1
        for c in chaifen:
            if len(c) == 3 and len(set(c)) == 1:
                if c[0] in sanyuan_dict:  # 三元牌刻子
                    fan += 1
                if c[0] == self.SelfWind:  # 自风刻子
                    fan += 1
                if c[0] == self.RoundWind:  # 场风刻子
                    fan += 1
        # 红宝牌
        for name in ['0m', '0p', '0s']:
            if name in self.screenInfo.handTiles:
                fan += 1
        # 宝牌
        for i in [str2id[name] for name in self.doraStack]:
            if self.handTiles[i] > 0:
                fan += self.handTiles[i]
        # 断幺
        duanyao = True
        for i in range(34):
            if idx2str[i] in yaojiu_dict and self.handTiles[i] != 0:
                duanyao = False
        if duanyao:
            fan += 1

        return fan

    # 防守
    def defend(self):
        safe_tiles = [[False] * 34, [False] * 34, [False] * 34, [False] * 34, ]
        syn_safe_tiles_0 = [False] * 34  # 绝对安全牌，所有立直方安全牌的交集
        syn_safe_tiles_1 = [False] * 34  # 对一家安全牌，所有立直方安全牌的并集
        syn_safe_tiles_2 = [False] * 34  # 相对安全牌：筋牌，字牌
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
                for tile in self.screenInfo.allPaiHeTiles_str[d][0] + self.screenInfo.allPaiHeTiles_str[d][1]:
                    safe_tiles[d][str2id[tile]] = True
                    # 筋牌
                    if tile[0] in ['4', '5', '6'] and tile[1] in ['m', 'p', 's']:
                        syn_safe_tiles_2[str2id['{}{}'.format(chr(ord(tile[0]) - 3), tile[1])]] = True
                        syn_safe_tiles_2[str2id['{}{}'.format(chr(ord(tile[0]) + 3), tile[1])]] = True
                for d2 in range(4):
                    if d2 != d:
                        tmp = self.screenInfo.allPaiHeTiles_str[d2][0] + self.screenInfo.allPaiHeTiles_str[d2][1]
                        tmp = tmp[liqi[d][1] + 1:]
                        for tile in tmp:
                            safe_tiles[d][str2id[tile]] = True
        for i in range(34):
            syn_safe_tiles_0[i] = (safe_tiles[1][i] or not liqi[1][0]) and (safe_tiles[2][i] or not liqi[2][0]) and (
                    safe_tiles[3][i] or not liqi[3][0])
            syn_safe_tiles_1[i] = (safe_tiles[1][i] and liqi[1][0]) or (safe_tiles[2][i] and liqi[2][0]) or (
                    safe_tiles[3][i] and liqi[3][0])

        # 相对安全的牌
        # 字牌
        for i in range(27, 34):
            syn_safe_tiles_2[i] = True

        return liqi[1][0] or liqi[2][0] or liqi[3][0], (syn_safe_tiles_0, syn_safe_tiles_1, syn_safe_tiles_2), \
               [l[0] for l in liqi]

    def getNextAction(self, lastDiscardTile=None):
        if len(set(self.screenInfo.handTiles).union(set(self.screenInfo.myFuLuTiles))) < 5:  # 输入有误
            print('error,', self.screenInfo.handTiles + self.myFuLuTiles)
            return None, None, None
        # 进攻
        # 优先级：最小向听数 > 最多进张数 > 最大价值
        # 首先满足最小向听数，然后在进张数的前3名中选择价值最大的
        if lastDiscardTile is not None:  # 吃/碰模式
            yis = self.Yi(ignore_menqing=not self.menqing)
            name, (dist, forced_guzhang_mask) = yis[0]  # 第一役种
            orig_shantin, _ = self.get_minshantin(forced_guzhang_mask)
            orig_jinzhang, _ = self.get_JinZhang(orig_shantin)

            tile_transform_addbit([lastDiscardTile], self.handTiles)

        records = []  # 记录所有选择

        for i in range(34):
            if self.handTiles[i] - self.myFuLuTiles[i] > 0:  # 不打已副露的牌
                self.handTiles[i] -= 1

                # forced_guzhang_mask：若mask某位为0，则手牌对应位必须拆解为孤张
                yis = self.Yi(ignore_menqing=not self.menqing or lastDiscardTile is not None)
                yi_infos = []
                for yi in yis:
                    name, (dist, forced_guzhang_mask) = yi
                    min_shantin, chaifen = self.get_minshantin(forced_guzhang_mask)

                    single_tile_values = self.calc_alldoras_value(idx2str[i]) + self.calc_alltiles_value()
                    maxValue = single_tile_values + self.get_maxvalue(forced_guzhang_mask)
                    max_jinzhang, names = self.get_JinZhang(min_shantin)

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
                            if forced_guzhang_mask[p] == 0 and idx2str[p] in names:
                                max_jinzhang -= names[idx2str[p]]
                                del names[idx2str[p]]
                        # 吃/碰
                        if lastDiscardTile is not None:
                            for c in chaifen:
                                flag = True  # c不能是副露
                                for p in c:
                                    if self.handTiles[str2id[p]] - self.myFuLuTiles[str2id[p]] == 0:
                                        flag = False

                                if flag and lastDiscardTile in c and len(c) == 3:
                                    choice = set(c) - {lastDiscardTile}
                                    # 将吃牌后的顺子加入到副露，重新计算向听数
                                    if is_shunzi(c):
                                        tile_transform_addbit(c, self.myFuLuTiles)
                                    else:  # 刻子
                                        tile_transform_addbit([lastDiscardTile], self.myFuLuTiles)
                                        tile_transform_addbit([lastDiscardTile], self.myFuLuTiles)
                                        tile_transform_addbit([lastDiscardTile], self.myFuLuTiles)

                                    min_shantin, chaifen = self.get_minshantin(forced_guzhang_mask)
                                    single_tile_values = self.calc_alldoras_value(idx2str[i]) + self.calc_alltiles_value()
                                    maxValue = single_tile_values + self.get_maxvalue(forced_guzhang_mask)
                                    max_jinzhang, names = self.get_JinZhang(min_shantin)

                                    if is_shunzi(c):
                                        tile_transform_subbit(c, self.myFuLuTiles)
                                    else:  # 刻子
                                        tile_transform_subbit([lastDiscardTile], self.myFuLuTiles)
                                        tile_transform_subbit([lastDiscardTile], self.myFuLuTiles)
                                        tile_transform_subbit([lastDiscardTile], self.myFuLuTiles)
                                    yi_infos.append(((name, dist), min_shantin, max_jinzhang,
                                                     maxValue, names, chaifen, choice))
                        else:
                            yi_infos.append(((name, dist), min_shantin, max_jinzhang,
                                             maxValue, names, chaifen, []))

                yi_infos = sorted(yi_infos, key=lambda x: x[0][1], reverse=False)  # 按距离升序
                if len(yi_infos) != 0:
                    records.append((idx2str[i], yi_infos))
                    # print(idx2str[i])
                    # self.print_records([records[-1]])

                self.handTiles[i] += 1

        min_dist = 100
        min_shantin = 8

        if DEBUG:
            print('flag1')
            self.print_records(records)

        # 第一役种最小向听预判断，先判断是否防守，决策放在后面
        for record in records:
            tile, yis = record
            (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
            if shantin < min_shantin:
                min_shantin = shantin

        # 防守判断，若有人立直并且自己向听数>=2，考虑弃和；否则跳过
        # 只打以下牌：立直方的牌河牌、立直巡数后的所有牌河牌、部分筋牌
        exist_liqi, syn_safe_tiles, liqi_condition = self.defend()
        absolutely_safe_tiles = []
        specific_safe_tiles = []
        maybe_safe_tiles = []
        if exist_liqi:
            for i in range(34):
                if syn_safe_tiles[0][i]:
                    absolutely_safe_tiles.append(idx2str[i])
                if syn_safe_tiles[1][i]:
                    specific_safe_tiles.append(idx2str[i])
                if syn_safe_tiles[2][i]:
                    maybe_safe_tiles.append(idx2str[i])

            print('liqi: ', liqi_condition)
        else:
            print('\033[0;32mno liqi\033[0m')

        if min_shantin <= 0:
            print('tingpai, ignore defense')
        elif exist_liqi:
            print('liqi defend applied')
            tmp = [record for record in records if record[0] in absolutely_safe_tiles]
            tmp2 = [record for record in records if record[0] in specific_safe_tiles]
            tmp3 = [record for record in records if record[0] in maybe_safe_tiles]
            if len(tmp) > 0:  # 有安全牌的打法，仍有进攻机会
                print('defend with attack')
                print_safe_tiles('absolutely_safe_tiles', absolutely_safe_tiles)
                records = tmp
            elif len(tmp2) > 0:
                print('partial defend with attack')
                print_safe_tiles('specific_safe_tiles', specific_safe_tiles)
                records = tmp2
            elif len(tmp3) > 0:
                print('prob_defend with attack')
                print_safe_tiles('maybe_safe_tiles', maybe_safe_tiles)
                records = tmp3
            else:  # 完全弃和，强制打出一张安全牌
                defend_tile = None
                for i in range(34):
                    if idx2str[i] in syn_safe_tiles and self.handTiles[i] - self.myFuLuTiles[i] > 0:
                        defend_tile = idx2str[i]
                        break
                if defend_tile is not None:
                    print('\033[0;31mdefend: {}, shantin={}\033[0m'.format(defend_tile, min_shantin))
                    if lastDiscardTile is not None:
                        tile_transform_subbit([lastDiscardTile], self.handTiles)
                    return defend_tile, None, False
                else:  # 手牌没有安全牌，放弃防守
                    print('no safe tiles, give up defense')

        safe_tiles = absolutely_safe_tiles.copy()
        if DEBUG:
            print('flag2')
            self.print_records(records, safe_tiles)
        # 选择第一距离最小打法，并且第一距离最小必须<=2
        for record in records:
            tile, yis = record
            if len(yis) > 0:
                (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
                if dist < min_dist:
                    min_dist = dist
        records = [record for record in records if record[1][0][0][1] == min_dist <= 2]
        if DEBUG:
            print('flag3')
            self.print_records(records)

        min_shantin = 8
        # 再次判断第一役种最小向听
        for record in records:
            tile, yis = record
            (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
            if shantin < min_shantin:
                min_shantin = shantin
        if lastDiscardTile is not None:
            records = [record for record in records if
                       record[1][0][1] == min_shantin and record[1][0][2] > 0 and record[1][0][1] <= orig_shantin
                       and (record[1][0][1] < orig_shantin or record[1][0][2] >= orig_jinzhang + 5)]
        else:
            records = [record for record in records if
                       record[1][0][1] == min_shantin and record[1][0][2] > 0]

        if DEBUG:
            print('flag4')
            self.print_records(records)

        records.sort(key=lambda record: record[1][0][3], reverse=True)  # 按价值降序
        if len(records) >= 6:  # 第一役种价值取前1/2
            records = records[:len(records) // 2 + 1]

        if DEBUG:
            print('flag5')
            self.print_records(records, safe_tiles)

        min_dist = 100
        # 第二距离最小，如果有第二距离且最小第二距离<=2的话
        for record in records:
            tile, yis = record
            if len(yis) > 1:
                (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[1]
                if dist < min_dist:
                    min_dist = dist
        if min_dist <= 2:
            records = [record for record in records if len(record[1]) < 2 or record[1][1][0][1] == min_dist]

        if DEBUG:
            print('flag6')
            self.print_records(records, safe_tiles)

        # 第一役种最多进张
        records.sort(key=lambda record: record[1][0][2], reverse=True)  # 按进张数降序排序
        records = records[:5]

        if DEBUG:
            print('flag7')
            self.print_records(records, safe_tiles)

        # 价值前4名
        records.sort(key=lambda record: record[1][0][3], reverse=True)  # 按价值降序
        syn_res = records[:4]
        # print('syn_res:', syn_res)
        if DEBUG:
            print('flag8')
            self.print_records(syn_res, safe_tiles)

        if min_shantin == 0:
            syn_res = self.actionTingPai(syn_res)

        if DEBUG:
            print('flag9')
            self.print_records(syn_res, safe_tiles)
        if len(syn_res) == 0:
            if lastDiscardTile is not None:
                tile_transform_subbit([lastDiscardTile], self.handTiles)
            return None, None, None

        syn_choice = syn_res[0]

        if lastDiscardTile is not None:
            tile, yis = syn_choice
            (name, dist), shantin, jinzhang, value, names, chaifen, choice = yis[0]
            if is_shunzi(choice.union({lastDiscardTile})):  # 吃牌后要弃的牌不能与另外两张牌组成顺子
                print(lastDiscardTile, choice)
                tile_transform_subbit([lastDiscardTile], self.handTiles)
                return None, None, None  # (tile, choice, liqi)
            print('chi/peng:', lastDiscardTile, tile, choice)
            print('shantin:{} to {}, jinzhang: {} to {}'.format(orig_shantin, shantin, orig_jinzhang, jinzhang))
            self.print_records([syn_choice], safe_tiles)
            return tile, choice, False  # (tile, choice, liqi)

        self.print_records([syn_choice], safe_tiles)
        return syn_choice[0], syn_choice[1][0][1], self.menqing and syn_choice[1][0][1] == 0  # (tile, shantin, liqi)

    def print_records(self, records, safe_tiles=None):
        if safe_tiles is None:
            safe_tiles = []
        for record in records:
            tile, yis = record
            print('\033[0;31mdiscard tile\033[0m: \033[0;31m{}\033[0m -> '.format(tile), end='')
            if tile in safe_tiles:
                print('\033[0;32msafe\033[0m')
            else:
                print('\033[0;31munsafe\033[0m')
            for yi in yis:
                (name, dist), min_shantin, max_jinzhang, maxValue, names, chaifen, choice = yi
                print('\t\033[0;32m{}\033[0m:\033[0;34m{}\033[0m, '.format(name, dist), end='')
                print('shantin:\033[0;34m{}\033[0m, '.format(min_shantin), end='')
                print('value:\033[0;34m{}\033[0m, '.format(maxValue), end='')
                print('jinzhang:\033[0;34m{}\033[0m->['.format(max_jinzhang), end='')
                for tile in names:
                    print('\033[0;34m{}\033[0m{} '.format(tile, names[tile]), end='')
                print('], chaifen:[', end='')
                for c in chaifen:
                    short = ''
                    for i in c:
                        short += i[0]
                    flag = c[0][1]
                    print('{}\033[0;34m{}\033[0m'.format(short, flag), end='')
                print(']')

    def is_guzhang(self, k):
        if k >= 27:
            return self.handTiles[k] <= 1
        res = False
        for i in range(-2, 3):
            if i != 0 and 0 <= k % 9 + i <= 8:
                res = res or self.handTiles[k + i] > 0
        return not res and self.handTiles[k] == 1

    def is_dora(self, k):
        return idx2str[k] in self.doraStack

    def calc_alldoras_value(self, discardTile):  # dora还给予了其相邻数牌额外价值
        res = 0
        doras = [str2id[name] for name in self.doraStack]
        for dname, name in [('0m', '5m'), ('0p', '5p'), ('0s', '5s')]:
            if dname in self.screenInfo.handTiles:  # 手牌中有红dora，如果弃的牌是唯一的5m，则不计红5m的价值
                if discardTile == name and self.handTiles[str2id[name]] != 0:  # 此时handTiles已经减1
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
                    res += self.handTiles[i] * TileValue[idx2str[i][0]]
                elif self.hiddenTiles[i] + self.handTiles[i] >= 3 and self.handTiles[i] + self.myFuLuTiles[i] <= 2:
                    val = 0
                    if 31 <= i <= 33:  # 场上出现张数越多，巡数越多价值越低，2张价值减为10
                        val = TileValue['sanyuan'] - 45 * self.allPaiHeTiles[i] - len(self.myPaiHe)
                    if str2id[self.screenInfo.SelfWind] == i:
                        val = TileValue['selfwind'] - 45 * self.allPaiHeTiles[i]
                    if str2id[self.screenInfo.RoundWind] == i:
                        val = TileValue['roundwind'] - 45 * self.allPaiHeTiles[i] - len(self.myPaiHe)
                    res += self.handTiles[i] * val
        return res
