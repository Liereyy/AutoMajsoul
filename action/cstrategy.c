#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

char* idx2str[] = {"1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
           "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
           "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
           "1z", "2z", "3z", "4z", "5z", "6z", "7z"};

// gcc -shared -o cstrategy.so -fPIC cstrategy.c

int handTiles[34];
int orig_handTiles[34];
int fuLuTiles[34];
int forcedGuZhangTiles[34];

int min_shantin;
int min_mianzi;
int min_dazi;
int min_quetou;
int chaifen_res[100];

int mianzi;
int dazi;
int quetou;

int mianzi_stack[10];
int mianzi_top = 0;

int kezi_stack[10];
int kezi_top = 0;

int dazi_stack[10];
int dazi_top = 0;

int dazi2_stack[10];
int dazi2_top = 0;

int quetou_stack[10];
int quetou_top = 0;

int guzhang_stack[20];
int guzhang_top = 0;

int get_XiangTingShu(int m, int dazi, int quetou)
{
    int d = dazi + quetou;
    int q;

    int c = m + d <= 5 ? 0 : m + d - 5;
    if (m + d <= 4)
        q = 1;
    else
        if (quetou)
            q = 1;
        else
            q = 0;
    return 9 - 2 * m - d + c - q;
}

int get_QiDUi_XiangTingShu(int quetou)
{
    int k = 0;
    for (int i = 0; i < 34; ++i)
        if (orig_handTiles[i] > 0)
            ++k;
    return 6 - quetou + fmax(0, 7 - k);
}

void save_chaifen()
{
    int k = 0;
    for (int i = 0; i < mianzi_top; ++i)
    {
        chaifen_res[k++] = mianzi_stack[i];
        chaifen_res[k++] = mianzi_stack[i]+1;
        chaifen_res[k++] = mianzi_stack[i]+2;
        chaifen_res[k++] = -2;
    }
    for (int i = 0; i < kezi_top; ++i)
    {
        chaifen_res[k++] = kezi_stack[i];
        chaifen_res[k++] = kezi_stack[i];
        chaifen_res[k++] = kezi_stack[i];
        chaifen_res[k++] = -2;
    }
    for (int i = 0; i < dazi_top; ++i)
    {
        chaifen_res[k++] = dazi_stack[i];
        chaifen_res[k++] = dazi_stack[i]+1;
        chaifen_res[k++] = -2;
    }
    for (int i = 0; i < dazi2_top; ++i)
    {
        chaifen_res[k++] = dazi2_stack[i];
        chaifen_res[k++] = dazi2_stack[i]+2;
        chaifen_res[k++] = -2;
    }
    for (int i = 0; i < quetou_top; ++i)
    {
        chaifen_res[k++] = quetou_stack[i];
        chaifen_res[k++] = quetou_stack[i];
        chaifen_res[k++] = -2;
    }
    for (int i = 0; i < guzhang_top; ++i)
    {
        int isGang = 0;
        for (int j = 0; j < kezi_top; ++j)
        {
            if (guzhang_stack[i] == kezi_stack[j])
                isGang = 1;
        }
        if (isGang) continue;
        
        chaifen_res[k++] = guzhang_stack[i];
        chaifen_res[k++] = -2;
    }
    if (k > 0)
        chaifen_res[k-1] = -3;
    else
        chaifen_res[0] = -3;
}

int max_value;

int value;

// value ???????????????????????????????????????????????????????????????
const int 
DaZi = 20,  // ????????????
DaZi2 = 10,  // ????????????
DaZi3 = 7,  // ????????????????????????(79)
DaZi4 = 6,  // ????????????????????????(89)
QueTou = 10,  // ??????
QueTou2 = 5,  // ?????????
QueTou3 = 7,  // ??????

// ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
AABBC1 = 100,  // AB?????????
AABBC2 = 120,  // AB?????????
AABCC1 = 100,
AABCC2 = 110,
ABBCC1 = 100,
ABBCC2 = 120,

ABCDE = 120,
ABCD = 90,
AABB1 = 15,
AABB2 = 60,

AAB1 = 20,
AAB2 = 30,
ABB1 = 20,
ABB2 = 30,

s1 = 1,
s2 = 2,
s3 = 3,
s4 = 4,
s5 = 5,
s6 = 4,
s7 = 3,
s8 = 2,
s9 = 1
;



void increase_mianzi(int k)
{
    --handTiles[k];
    --handTiles[k+1];
    --handTiles[k+2];
    ++mianzi;
    mianzi_stack[mianzi_top++] = k;
}

void decrease_mianzi(int k)
{
    ++handTiles[k];
    ++handTiles[k+1];
    ++handTiles[k+2];
    --mianzi;
    --mianzi_top;
}

void increase_quetou(int k)
{
    handTiles[k] -= 2;
    ++quetou;
    quetou_stack[quetou_top++] = k;
    if (k <= 26)
    {
        if (k % 9 == 0 || k % 9 == 8)
            value += QueTou2;
        else
            value += QueTou;
    }
    else
        value += QueTou3;
}

void decrease_quetou(int k)
{
    handTiles[k] += 2;
    --quetou;
    --quetou_top;
    if (k <= 26)
    {
        if (k % 9 == 0 || k % 9 == 8)
            value -= QueTou2;
        else
            value -= QueTou;
    }
    else
        value -= QueTou3;
}

void increase_anke(int k)
{
    handTiles[k] -= 3;
    ++mianzi;
    kezi_stack[kezi_top++] = k;
}

void decrease_anke(int k)
{
    handTiles[k] += 3;
    --mianzi;
    --kezi_top;
}

void increase_dazi(int k) // ??????/????????????
{
    --handTiles[k];
    --handTiles[k+1];
    ++dazi;
    dazi_stack[dazi_top++] = k;
    if (k % 9 >= 1 && k % 9 <= 6)
        value += DaZi;
    else
        value += DaZi4;
}

void decrease_dazi(int k)
{
    ++handTiles[k];
    ++handTiles[k+1];
    --dazi;
    --dazi_top;
    if (k % 9 >= 1 && k % 9 <= 6)
        value -= DaZi;
    else
        value -= DaZi4;
}

void increase_dazi2(int k)
{
    --handTiles[k];
    --handTiles[k+2];
    ++dazi;
    dazi2_stack[dazi2_top++] = k;
    if (k % 9 >= 1 && k % 9 <= 5)
        value += DaZi2;
    else
        value += DaZi3;
}

void decrease_dazi2(int k)
{
    ++handTiles[k];
    ++handTiles[k+2];
    --dazi;
    --dazi2_top;
    if (k % 9 >= 1 && k % 9 <= 5)
        value -= DaZi2;
    else
        value -= DaZi3;
}

void increase_guzhang(int k)
{
    --handTiles[k];
    guzhang_stack[guzhang_top++] = k;
}

void decrease_guzhang(int k)
{
    ++handTiles[k];
    --guzhang_top;
}

// ?????????
void increaseAABBC(int k)
{
    handTiles[k] -= 2;
    handTiles[k + 1] -= 2;
    handTiles[k + 2] -= 1;
    if (k % 9 >= 1 && k % 9 <= 6)
        value += AABBC2;
    else
        value += AABBC1;
}

void decreaseAABBC(int k)
{
    handTiles[k] += 2;
    handTiles[k + 1] += 2;
    handTiles[k + 2] += 1;
    if (k % 9 >= 1 && k % 9 <= 6)
        value -= AABBC2;
    else
        value -= AABBC1;
}

void increaseAABCC(int k)
{
    handTiles[k] -= 2;
    handTiles[k + 1] -= 1;
    handTiles[k + 2] -= 2;
    if (k % 9 >= 1 && k % 9 <= 5)
        value += AABCC2;
    else
        value += AABCC1;
}

void decreaseAABCC(int k)
{
    handTiles[k] += 2;
    handTiles[k + 1] += 1;
    handTiles[k + 2] += 2;
    if (k % 9 >= 1 && k % 9 <= 5)
        value -= AABCC2;
    else
        value -= AABCC1;
}

void increaseABBCC(int k)
{
    handTiles[k] -= 1;
    handTiles[k + 1] -= 2;
    handTiles[k + 2] -= 2;
    if (k % 9 >= 0 && k % 9 <= 5)
        value += ABBCC2;
    else
        value += ABBCC1;
}

void decreaseABBCC(int k)
{
    handTiles[k] += 1;
    handTiles[k + 1] += 2;
    handTiles[k + 2] += 2;
    if (k % 9 >= 0 && k % 9 <= 5)
        value -= ABBCC2;
    else
        value -= ABBCC1;
}

void increaseAAB(int k)
{
    handTiles[k] -= 2;
    handTiles[k + 1] -= 1;
    if (k % 9 >= 1 && k % 9 <= 6)
        value += AAB2;
    else
        value += AAB1;
}

void decreaseAAB(int k)
{
    handTiles[k] += 2;
    handTiles[k + 1] += 1;
    if (k % 9 >= 1 && k % 9 <= 6)
        value -= AAB2;
    else
        value -= AAB1;
}

void increaseABB(int k)
{
    handTiles[k] -= 1;
    handTiles[k + 1] -= 2;
    if (k % 9 >= 1 && k % 9 <= 6)
        value += ABB2;
    else
        value += ABB1;
}

void decreaseABB(int k)
{
    handTiles[k] += 1;
    handTiles[k + 1] += 2;
    if (k % 9 >= 1 && k % 9 <= 6)
        value -= ABB2;
    else
        value -= ABB1;
}

void increaseAABB(int k)
{
    handTiles[k] -= 2;
    handTiles[k + 1] -= 2;
    if (k % 9 >= 1 && k % 9 <= 6)
        value += AABB2;
    else
        value += AABB1;
}

void decreaseAABB(int k)
{
    handTiles[k] += 2;
    handTiles[k + 1] += 2;
    if (k % 9 >= 1 && k % 9 <= 6)
        value -= AABB2;
    else
        value -= AABB1;
}

void increaseABCD(int k)
{
    handTiles[k] -= 1;
    handTiles[k + 1] -= 1;
    handTiles[k + 2] -= 1;
    handTiles[k + 3] -= 1;
    value += ABCD;
}

void decreaseABCD(int k)
{
    handTiles[k] += 1;
    handTiles[k + 1] += 1;
    handTiles[k + 2] += 1;
    handTiles[k + 3] += 1;
    value -= ABCD;
}

void increaseABCDE(int k)
{
    handTiles[k] -= 1;
    handTiles[k + 1] -= 1;
    handTiles[k + 2] -= 1;
    handTiles[k + 3] -= 1;
    handTiles[k + 4] -= 1;
    value += ABCDE;
}

void decreaseABCDE(int k)
{
    handTiles[k] += 1;
    handTiles[k + 1] += 1;
    handTiles[k + 2] += 1;
    handTiles[k + 3] += 1;
    handTiles[k + 4] += 1;
    value -= ABCDE;
}

void fulu_pre_fix(int depth)
{
    if (depth >= 34)
        return;

    if (fuLuTiles[depth] == 0)
        fulu_pre_fix(depth + 1);
    else if (fuLuTiles[depth] >= 3)
    {
        increase_anke(depth);
        fuLuTiles[depth] -= 3;
        fulu_pre_fix(depth);
    }
    else if (depth <= 26)
    {
        int i = depth;
        i %= 9;

        if (fuLuTiles[depth] >= 1)
            if (i <= 6 && fuLuTiles[depth+1] > 0 && fuLuTiles[depth+2] > 0)
            {
                increase_mianzi(depth);
                --fuLuTiles[depth];
                --fuLuTiles[depth+1];
                --fuLuTiles[depth+2];
                fulu_pre_fix(depth);
            }
    }
}

void forcedGuZhang_pre()
{
    for (int i = 0; i < 34; ++i)
        if (forcedGuZhangTiles[i] == 0)
            while (handTiles[i] > 0)
                increase_guzhang(i);
}

void init(char* handTileCode, char* myFuLuTileCode, char* forcedGuZhangMask)
{
    mianzi = 0, dazi = 0, quetou = 0, min_shantin = 8;
    max_value = -1, value = 0;
    mianzi_top = 0, kezi_top = 0, dazi_top = 0, dazi2_top = 0, quetou_top = 0, guzhang_top = 0;

    for (int i = 0; i < 34; ++i)
    {
        // ????????????????????????????????????handTileCode????????????34
        if (i < strlen(handTileCode))
            handTiles[i] = orig_handTiles[i] = handTileCode[i] - '0';
        else
            handTiles[i] = orig_handTiles[i] = 0;
        
        if (i < strlen(myFuLuTileCode))
            fuLuTiles[i] = myFuLuTileCode[i] - '0';
        else
            fuLuTiles[i] = 0;
        
        if (i < strlen(forcedGuZhangMask))
            forcedGuZhangTiles[i] = forcedGuZhangMask[i] - '0';
        else
            forcedGuZhangTiles[i] = 0;
    }


    // ?????????
    forcedGuZhang_pre();  // ???????????????
    fulu_pre_fix(0);  // ???????????????
}

void shantin_aux(int depth)
{
    if (depth >= 34)
    {
        int shantin1 = get_XiangTingShu(mianzi, dazi, quetou);
        int shantin = shantin1;
        int shantin2 = get_QiDUi_XiangTingShu(quetou);
        if (shantin2 <= 3 && shantin2 < shantin1)  // ???????????????<=3??????????????????
            shantin = shantin1;

        if (shantin < min_shantin)
        {
            min_mianzi = mianzi;
            min_dazi = dazi;
            min_quetou = quetou;
            min_shantin = shantin;
            save_chaifen();
        }
        return;
    }

    if (handTiles[depth] == 0)
        shantin_aux(depth + 1);
    else if (depth <= 26)
    {
        int i = depth;
        i %= 9;

        if (handTiles[depth] >= 1)
        {
            increase_guzhang(depth);
            shantin_aux(depth);
            decrease_guzhang(depth);

            if (i <= 6 && handTiles[depth+1] > 0 && handTiles[depth+2] > 0)
            {
                increase_mianzi(depth);
                shantin_aux(depth);
                decrease_mianzi(depth);
            }
            if (i <= 7 && handTiles[depth+1] > 0)
            {
                increase_dazi(depth);
                shantin_aux(depth);
                decrease_dazi(depth);
            }
            if (i <= 6 && handTiles[depth+2] > 0)
            {
                increase_dazi2(depth);
                shantin_aux(depth);
                decrease_dazi2(depth);
            }
        }

        if (handTiles[depth] >= 2)
        {
            increase_quetou(depth);
            shantin_aux(depth);
            decrease_quetou(depth);
        }

        if (handTiles[depth] >= 3)
        {
            increase_anke(depth);
            shantin_aux(depth);
            decrease_anke(depth);
        }
    }
    else
    {
        if (handTiles[depth] >= 1)
        {
            increase_guzhang(depth);
            shantin_aux(depth);
            decrease_guzhang(depth);
        }
        if (handTiles[depth] >= 2)
        {
            increase_quetou(depth);
            shantin_aux(depth);
            decrease_quetou(depth);
        }
        if (handTiles[depth] >= 3)
        {
            increase_anke(depth);
            shantin_aux(depth);
            decrease_anke(depth);
        }
    }
}

void value_aux(int depth)
{
    if (depth >= 34)
    {
        if (value > max_value)
        {
            max_value = value;
        }
        return;
    }

    if (handTiles[depth] == 0)
        value_aux(depth + 1);
    else if (depth <= 26)
    {
        // m/p/s
        int i = depth;
        i %= 9;

        if (handTiles[depth] >= 1)
        {
            if (i <= 6 && handTiles[depth + 1] >= 2 && handTiles[depth + 2] >= 2)
            {
                increaseABBCC(depth);
                value_aux(depth);
                decreaseABBCC(depth);
            }
            if (i >= 1 && i <= 3 && handTiles[depth + 1] > 0 && handTiles[depth + 2] > 0 
                    && handTiles[depth + 3] > 0)
            {
                increaseABCDE(depth);
                value_aux(depth);
                decreaseABCDE(depth);
            }
            if (i <= 6 && handTiles[depth + 1] > 0 && handTiles[depth + 2] > 0)
            {
                increase_mianzi(depth);
                value_aux(depth);
                decrease_mianzi(depth);
            }
            if (i <= 6 && handTiles[depth + 1] >= 2)
            {
                increaseABB(depth);
                value_aux(depth);
                decreaseABB(depth);
            }
            // ??????
            increase_guzhang(depth);
            value_aux(depth);
            decrease_guzhang(depth);

            if (i <= 7 && handTiles[depth + 1] > 0)
            {
                // ????????????
                increase_dazi(depth);
                value_aux(depth);
                decrease_dazi(depth);
            }
            if (i <= 6 && handTiles[depth + 2] > 0)
            {
                // ????????????
                increase_dazi2(depth);
                value_aux(depth);
                decrease_dazi2(depth);
            }
        }

        if (handTiles[depth] >= 2)
        {
            increase_quetou(depth);
            value_aux(depth);
            decrease_quetou(depth);

            if (i <= 6 && handTiles[depth + 1] >= 1 && handTiles[depth + 2] >= 2)
            {
                increaseAABCC(depth);
                value_aux(depth);
                decreaseAABCC(depth);
            }
            if (i <= 7 && handTiles[depth + 1] >= 1)
            {
                increaseAAB(depth);
                value_aux(depth);
                decreaseAAB(depth);
            }
            if (i <= 7 && handTiles[depth + 1] >= 2)
            {
                increaseAABB(depth);
                value_aux(depth);
                decreaseAABB(depth);
            }
        }

        if (handTiles[depth] >= 3)
        {
            increase_anke(depth);
            value_aux(depth);
            decrease_anke(depth);
        }
    }
    else
    {
        // ??????
        if (handTiles[depth] >= 1)
        {
            increase_guzhang(depth);
            value_aux(depth);
            decrease_guzhang(depth);
        }
        if (handTiles[depth] >= 2)
        {
            increase_quetou(depth);
            value_aux(depth);
            decrease_quetou(depth);
        }
        if (handTiles[depth] >= 3)
        {
            increase_anke(depth);
            value_aux(depth);
            decrease_anke(depth);
        }
    }
}

char res[100];
char* get_shantin(char* hanTileCode, char* myFuLuTileCode, char* forcedGuZhangMask)
{
    // printf("%s\n", s);
    init(hanTileCode, myFuLuTileCode, forcedGuZhangMask);
    shantin_aux(0);
    // printf("cshantin:%s, %d,%d,%d, %d\n", s, min_mianzi, min_dazi, min_quetou, min_shantin);
    int k = 0;
    res[k++] = min_shantin + '0';
    res[k++] = -2 + '0';
    while (chaifen_res[k-2] != -3)
    {
        res[k] = chaifen_res[k-2] + '0';
        ++k;
    }
    res[k] = '\0';
    // printf("cres:%s\n", res);
    return res;
}

int get_value(char* hanTileCode, char* myFuLuTileCode, char* forcedGuZhangMask)
{
    init(hanTileCode, myFuLuTileCode, forcedGuZhangMask);
    value_aux(0);
    return max_value;
}
