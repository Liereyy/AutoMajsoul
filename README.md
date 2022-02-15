# Strategy

## 框架

```python
yis = self.Yi(ignore_menqing=not self.menqing or lastDiscardTile is not None)
yi_infos = []
for yi in yis:
    name, (dist, forced_guzhang_mask) = yi
    min_shantin, chaifen = self.get_minshantin(forced_guzhang_mask)

    single_tile_values = self.calc_alldoras_value() + self.calc_alltiles_value()
    maxValue = single_tile_values + self.get_maxvalue(forced_guzhang_mask)

    max_jinzhang, names = self.get_JinZhang(min_shantin)
    ...
    yi_infos.append(
        ((name, dist), min_shantin, max_jinzhang, maxValue, names, chaifen, choice)
    )
records.append(
    	(idx2str[i], yi_infos)
)
```

## 决策优先级

#### stage0(pre_stage): 预计算第一役种最小向听，以决定是否防守

如果有人立直且自己向听数$\ge1$，则只保留是安全牌的打法。

如果没有安全牌的打法但手牌有安全牌，则完全弃和，强制打出一张安全牌。

如果手牌中没有安全牌，则考虑一般而言相对安全的牌。

若以上条件都不满足，放弃防守。

#### stage1: 第一役种最小距离

如果是门清，认为有门清役，records距离全为0；如果要吃牌，或者已有副露，则需确定与役种集中距离最小的役种名和距离。距离定义：为满足役需要弃的最少牌数。

#### stage2: 第一役种最小向听

保留向听数最小的打法，且进张数需$>0$

如果是吃/碰模式，还需吃后满足其一：1. 第一役种向听数$<$原手牌向听数，2. 相等但第一役种进张数显著增加(>=5)

#### stage3: 按第一役种价值降序取前1/3

#### stage4: 第二役种最小距离（如果有第二距离且最小第二距离<=2的话）

#### stage5: 第一役种最多进张

#### stage6: 第一役种最大价值

#### 此时得到的records集即为最终结果



## 役种集

1. 门清役（立直）
2. 役牌：白、中、发、自风、场风
3. 断幺九
4. 对对和
5. 混（清）一色
6. 三色同顺
7. 混（清）老头