from strategy import *


def getHandTiles(str):
    res = []
    k = 0
    for i in range(len(str)):
        if str[i] == 'm':
            for j in range(k, i):
                res.append('{}m'.format(str[j]))
            k = i + 1
        if str[i] == 'p':
            for j in range(k, i):
                res.append('{}p'.format(str[j]))
            k = i + 1
        if str[i] == 's':
            for j in range(k, i):
                res.append('{}s'.format(str[j]))
            k = i + 1
        if str[i] == 'z':
            for j in range(k, i):
                res.append('{}z'.format(str[j]))
            k = i + 1
    return [(i, (0, 0)) for i in res]


handTiles = getHandTiles('')
print(handTiles)
yiPai = ['1z', '5z', '6z', '7z']
screen_info = ScreenInfo('1z', '1z', [], handTiles, [], [], [], [], yiPai)
strategy = Strategy(screen_info)

for i in range(34):
    if strategy.handTiles[i] > 0:
        strategy.handTiles[i] -= 1

        min_shantin, chaifen = strategy.get_minshantin(MASK_FLAG)

        max_jinzhang, names = strategy.get_JinZhang(min_shantin, MASK_FLAG)

        print(id2str[i], names)

        strategy.handTiles[i] += 1

print(strategy.getNextAction())