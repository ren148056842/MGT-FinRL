# 图实现只读核查

生成代码：share_holding_graph.ipynb cells 1—21；预测使用图见GNNtransformerbankclosenew-8RUNS消融_FIXED.ipynb cells9、18、20。

共同股东集合筛选后，配对边权增加(ratio_i+ratio_j)/200，并对称累加；这不是乘积/余弦公式。股东名候选从data.iloc[j,i+2]枚举，实际匹配data2.iloc[j,k+1]、比率data2.iloc[j,k+11]，位置列契约缺失；未读取不存在的stockholder.csv，也未将该偏移风险直接声称为已证实唯一错误。可视化内嵌矩阵存在非零对角，需核对实体重复、输入schema及实际NPY的生成关系。图保持不变，来源和实质正确性仍待认证。

实际图顺序[KD,QA,SH]；A6/A7/A8分别去KD/QA/SH。WM是可训练的基础邻接矩阵，与掩码加权的固定图相加。关闭一个固定图不意味着去掉该关系的所有可学习通路，也无法隔离图的因果贡献。未做无WM/等参数/图重训对照。

逐股票Transformer沿时间轴attention；未添加跨股票attention。股东图公开披露时点、QA语料截止、成交量Kendall窗口截止均不可认证；本轮未测预测器训练/推理耗时。回测接口耗时不能充当预测开销。
