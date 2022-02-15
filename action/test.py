import ctypes

lib = ctypes.cdll.LoadLibrary("D:/github_workspace/codes/majsoul/action/cstrategy.so")
handTiles = '0011111101101000002000001000001000'.encode()
fuluTiles = '0000000000000000000000000000000000'.encode()
mask = '1111111111111111111111111111111111'.encode()
lib.get_shantin.restype = ctypes.c_char_p
# res = lib.get_shantin(handTiles, fuluTiles, mask).decode()
res = lib.get_value(handTiles, fuluTiles, mask)
# res = [[ord(i) - ord('0') for i in r] for r in res.split('.')]
print(res)
# tiles = ['5m', '3m']
# tiles = sorted(tiles, key=lambda x: x[0])
# print(tiles)

# mask_base = [i for i in range(34)]
# lst = [[1 if (i <= 26 and (p <= i <= p + 2 or p+9 <= i <= p+9 + 2 or p+18 <= i <= p+18 + 2))
#              or i >= 27 else 0 for i in mask_base]for p in range(7)]
# l2 = [[1 if (i <= 26 and 9 * k <= i <= 9 * k + 8) or i >= 27 else 0 for i in mask_base] for k in range(3)]
# for l in l2:
#     print(l)