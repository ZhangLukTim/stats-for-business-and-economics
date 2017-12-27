# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from scipy.stats import norm, t

# 配置plt
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

DIR = '../数据文件/第8章'
fname = 'Professional.xlsx'

cols_pair = {
    'Age': 'age',
    'Gender': 'gender',
    'Real Estate Purchases?': 'real_estate',
    'Value of Investments ($)': 'investment',
    'Number of Transactions': 'transaction',
    'Broadband Access?': 'hasBroadband',
    'Household Income ($)': 'income',
    'Have Children?': 'hasChildren'
}

data = pd.read_excel(os.path.join(DIR, fname)).iloc[:, :8]
data.rename(columns=cols_pair, inplace=True)

cls_feats = data.dtypes[data.dtypes == object]
num_feats = data.dtypes[data.dtypes != object]

# 对字符属性进行编码
for feat in cls_feats.index:
    le = preprocessing.LabelEncoder()
    data[feat] = le.fit_transform(data[feat])

# Q1. 利用适当的描述统计量对数据进行汇总
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(len(num_feats)):
    x, y = i // 2, i % 2
    _ax = ax[x][y]
    sns.distplot(data[num_feats.index[i]], ax=_ax)
    # _ax.set(title=num_feats.index[i])
plt.show()

# Q2. 建立订户平均年龄和平均家庭收入的95%的置信区间
def confidence_interval(data, confidence, target=None):
    """ 计算置信区间
    当target==None时，表示求data均值的置信区间，用t分布
    当指定了target，表示求target在data中的比例的置信区间，用正态分布
    * 当data为分类数据时，假设data的值只有2类( 0/1, Yes/No, ...)

    Params
    --------
    data: ndarray, 样本数据
    confidence: 置信度

    Return
    -------
    返回置信区间
    """

    n = len(data)
    alpha = (1 - confidence) / 2

    if target is None:
        # 均值, 标准差
        mean, std = np.mean(data), np.std(data)

        # 计算在自由度为n-1的t分布上侧面积为alpha / 2时的t值
        t_alpha = t.isf(alpha, df=n-1)
        E = t_alpha * ( std / np.sqrt(n) )
        interval = (mean - E, mean + E)
    else:
        # 计算目标值的比例
        p = np.mean(data==target)

        # 计算标准正态分布上侧面积为alpha / 2时的z值
        z_alpha = norm.isf(alpha)
        E = z_alpha * np.sqrt( p * (1 - p) / n)
        interval = (p - E, p + E)
    return interval

# 计算置信区间
age_interval = confidence_interval(data['age'], 0.95)
income_interval = confidence_interval(data['income'], 0.95)

# Q3. 建立订户家中有宽带接入和有子女的比率的95%的置信区间
broadband_interval = confidence_interval(data['hasBroadband'], 0.95, target=1)
children_interval = confidence_interval(data['hasChildren'], 0.95, target=1)

# Q4. 对在线代理商而言，Young Professional 杂志是一个好的代理渠道吗？
#         根据统计数据判断你的结论


# Q5. 对销售儿童教育软件和儿童计算机游戏的公司来说，该杂志是刊登广告的好地方吗？

# Q6. 读者会对哪种类型的文章感兴趣
