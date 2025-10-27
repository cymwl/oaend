# pcc.py
import pandas as pd
import numpy as np


def PCC(Y, X, num):
    """
    皮尔逊相关性系数分析

    参数:
    Y - 目标变量 (重金属含量), pandas Series
    X - 光谱数据, pandas DataFrame
    num - 要保留的特征数量

    返回:
    X_selected - 选择后的特征光谱数据
    corr - 相关系数 (不返回给前端)
    """
    # 确保输入格式正确
    if not isinstance(Y, pd.Series):
        Y = pd.Series(Y)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    Y = Y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    # 合并数据
    All = pd.concat([Y, X], axis=1)
    # All.columns = ['Target'] + list(X.columns)
    data = pd.DataFrame(All)
    corr = data.corr()
    corr_Y = corr.iloc[0, 1:]
    corr_absY = abs(corr_Y)
    lar_c = corr_absY.nlargest(num).index
    x = X[lar_c]
    corr = corr.filter(items=lar_c, axis=1).iloc[0, :]
    # 计算相关系数
    # corr = All.corr()
    # corr_Y = corr.iloc[0, 1:]  # 第一行是目标变量与其他特征的相关系数
    # corr_absY = abs(corr_Y)

    # 选择相关性最高的特征
    # top_features = corr_absY.nlargest(num).index
    # x = X[top_features]
    # return X_selected, corr_Y

    return x, corr