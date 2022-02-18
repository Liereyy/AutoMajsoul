from enum import Enum

import cv2


class Operation(Enum):
    NoEffect = 0
    Ordinary = 1
    Chi = 2
    Peng = 3


op2str = {Operation.NoEffect: 'NoEffect', Operation.Ordinary: 'Ordinary', Operation.Chi: 'Chi',
          Operation.Peng: 'Peng', }


def cv_show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_XiangTingShu(mianzi, dazi, quetou):
    m = mianzi
    d = dazi + quetou

    c = 0 if m + d <= 5 else m + d - 5
    if m + d <= 4:
        q = 1
    else:
        if quetou:
            q = 1
        else:
            q = 0
    return 9 - 2 * m - d + c - q


yaojiu_dict = ['1m', '9m', '1p', '9p', '1s', '9s', '1z', '2z', '3z', '4z', '5z', '6z', '7z']
laotou_dict = ['1m', '9m', '1p', '9p', '1s', '9s']

m_dict = ['0m', '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m']
p_dict = ['0p', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p']
s_dict = ['0s', '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s']
z_dict = ['1z', '2z', '3z', '4z', '5z', '6z', '7z']
feng_dict = ['1z', '2z', '3z', '4z']
sanyuan_dict = ['5z', '6z', '7z']

id2str = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
          '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
          '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
          '1z', '2z', '3z', '4z', '5z', '6z', '7z']

str2id = {
    '1m': 0, '2m': 1, '3m': 2, '4m': 3, '0m': 4, '5m': 4, '6m': 5, '7m': 6, '8m': 7, '9m': 8,
    '1p': 9, '2p': 10, '3p': 11, '4p': 12, '0p': 13, '5p': 13, '6p': 14, '7p': 15, '8p': 16, '9p': 17,
    '1s': 18, '2s': 19, '3s': 20, '4s': 21, '0s': 22, '5s': 22, '6s': 23, '7s': 24, '8s': 25, '9s': 26,
    '1z': 27, '2z': 28, '3z': 29, '4z': 30, '5z': 31, '6z': 32, '7z': 33,
}

name2id = {
    '0m': 0, '1m': 1, '2m': 2, '3m': 3, '4m': 4, '5m': 5, '6m': 6, '7m': 7, '8m': 8, '9m': 9,
    '0p': 10, '1p': 11, '2p': 12, '3p': 13, '4p': 14, '5p': 15, '6p': 16, '7p': 17, '8p': 18, '9p': 19,
    '0s': 20, '1s': 21, '2s': 22, '3s': 23, '4s': 24, '5s': 25, '6s': 26, '7s': 27, '8s': 28, '9s': 29,
    '1z': 30, '2z': 31, '3z': 32, '4z': 33, '5z': 34, '6z': 35, '7z': 36,
}


def tile_transform_addbit1(raw, array):
    for tile in raw:
        if tile in m_dict:
            if tile[0] == '0':
                array[4] += 1
            else:
                array[int(tile[0]) - 1] += 1
        if tile in p_dict:
            if tile[0] == '0':
                array[13] += 1
            else:
                array[9 + int(tile[0]) - 1] += 1
        if tile in s_dict:
            if tile[0] == '0':
                array[22] += 1
            else:
                array[18 + int(tile[0]) - 1] += 1
        if tile in feng_dict or tile in sanyuan_dict:
            array[27 + int(tile[0]) - 1] += 1


def tile_transform_addbit2(raw, array):
    for tile, rotated in raw:
        if tile in m_dict:
            if tile[0] == '0':
                array[4] += 1
            else:
                array[int(tile[0]) - 1] += 1
        if tile in p_dict:
            if tile[0] == '0':
                array[13] += 1
            else:
                array[9 + int(tile[0]) - 1] += 1
        if tile in s_dict:
            if tile[0] == '0':
                array[22] += 1
            else:
                array[18 + int(tile[0]) - 1] += 1
        if tile in feng_dict or tile in sanyuan_dict:
            array[27 + int(tile[0]) - 1] += 1


def tile_transform_subbit1(raw, array):
    for tile in raw:
        if tile in m_dict:
            if tile[0] == '0':
                array[4] -= 1
            else:
                array[int(tile[0]) - 1] -= 1
        if tile in p_dict:
            if tile[0] == '0':
                array[13] -= 1
            else:
                array[9 + int(tile[0]) - 1] -= 1
        if tile in s_dict:
            if tile[0] == '0':
                array[22] -= 1
            else:
                array[18 + int(tile[0]) - 1] -= 1
        if tile in feng_dict or tile in sanyuan_dict:
            array[27 + int(tile[0]) - 1] -= 1


def tile_transform_subbit2(raw, array):
    for tile, rotated in raw:
        if tile in m_dict:
            if tile[0] == '0':
                array[4] -= 1
            else:
                array[int(tile[0]) - 1] -= 1
        if tile in p_dict:
            if tile[0] == '0':
                array[13] -= 1
            else:
                array[9 + int(tile[0]) - 1] -= 1
        if tile in s_dict:
            if tile[0] == '0':
                array[22] -= 1
            else:
                array[18 + int(tile[0]) - 1] -= 1
        if tile in feng_dict or tile in sanyuan_dict:
            array[27 + int(tile[0]) - 1] -= 1


def tile_transform_stack(raw, stack):
    for dora in raw:
        if dora in m_dict:
            if dora[0] == '0':
                stack.append('{}m'.format(6))
            else:
                stack.append('{}m'.format(int(dora[0]) % 9 + 1))
        if dora in p_dict:
            if dora[0] == '0':
                stack.append('{}p'.format(6))
            else:
                stack.append('{}p'.format(int(dora[0]) % 9 + 1))
        if dora in s_dict:
            if dora[0] == '0':
                stack.append('{}s'.format(6))
            else:
                stack.append('{}s'.format(int(dora[0]) % 9 + 1))
        if dora in feng_dict:
            stack.append('{}z'.format(int(dora[0]) % 4 + 1))
        if dora in sanyuan_dict:
            stack.append('{}z'.format((int(dora[0]) - 4) % 3 + 5))


def dist_to_side(tile):
    if tile[0] == '0':
        return 4
    if tile in m_dict or tile in p_dict or tile in s_dict:
        return min(int(tile[0]) - 1, 9 - int(tile[0])) // 2
    return 0


def print_tiles(msg, tiles, end='\n'):
    m = ''
    p = ''
    s = ''
    z = ''
    for tile in tiles:
        if tile[1] == 'm':
            m += tile[0]
        if tile[1] == 'p':
            p += tile[0]
        if tile[1] == 's':
            s += tile[0]
        if tile[1] == 'z':
            z += tile[0]

    print('\033[0;32m{}: \033[0m'.format(msg), end='')
    if len(m) != 0:
        print('\033[0;32m{}\033[0m\033[0;34mm\033[0m'.format(m), end='')
    if len(p) != 0:
        print('\033[0;32m{}\033[0m\033[0;34mp\033[0m'.format(p), end='')
    if len(s) != 0:
        print('\033[0;32m{}\033[0m\033[0;34ms\033[0m'.format(s), end='')
    if len(z) != 0:
        print('\033[0;32m{}\033[0m\033[0;34mz\033[0m'.format(z), end='')
    print(end, end='')


def is_shunzi(tiles):
    if len(tiles) != 3:
        return False
    tmp = []
    n = []
    for tile in tiles:
        tmp.append(tile[1])
        n.append(int(tile[0]))
    if len(set(tmp)) == 1 and tmp[0] != 'z':  # 全部是一色且不是字牌
        n.sort()
        if n[2] == n[1] + 1 == n[0] + 2:
            return True
    return False


def tiles_encode(tiles):
    tileCode = ''
    for k in tiles:
        tileCode += str(k)
    return tileCode.encode()


def print_records(records, _safe_tiles=None):
    if _safe_tiles is None:
        _safe_tiles = []
    for record in records:
        tile, yis, op = record
        print('\033[0;31mdiscard tile\033[0m: \033[0;31m{}\033[0m {} -> '.format(tile, op2str[op]), end='')
        for safe_tiles, name in _safe_tiles:
            if tile in safe_tiles:
                print('{}: \033[0;32msafe\033[0m'.format(name), end='  ')
                break
            else:
                print('{}: \033[0;31munsafe\033[0m'.format(name), end='  ')
        print()
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
