mask_base = [i for i in range(34)]
MASK_FLAG = [1] * 34  # 所有手牌都可以存在

mMask = [1 if 0 <= i <= 8 else 0 for i in mask_base]
pMask = [1 if 9 <= i <= 17 else 0 for i in mask_base]
sMask = [1 if 18 <= i <= 26 else 0 for i in mask_base]
mzMask = [1 if 0 <= i <= 8 or 27 <= i <= 33 else 0 for i in mask_base]
pzMask = [1 if 9 <= i <= 17 or 27 <= i <= 33 else 0 for i in mask_base]
szMask = [1 if 18 <= i <= 26 or 27 <= i <= 33 else 0 for i in mask_base]
zMask = [1 if 27 <= i <= 33 else 0 for i in mask_base]
tMask = mMask, pMask, sMask, zMask
tzMask = mzMask, pzMask, szMask, zMask

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