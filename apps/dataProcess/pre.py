import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import pandas as pd
from scipy.interpolate import interp1d
# from Preprocessing import Sppre
import pywt
from sklearn.preprocessing import StandardScaler


# 小波变换
def WAVE(data):  # 小波变换的输入date必须是偶数列
    labels = data.columns

    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    else:
        data = data.copy()

    # if data.shape[1] % 2 != 0:
    #     data_values = data[:, :-1]
    #     print(f"警告：输入数据列数为奇数({data_values.shape[1] + 1})，已自动截断为偶数列({data_values.shape[1]})")
    #     # 同时调整标签
    #     original_labels = labels[:-1]
    def wave_(data):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data.shape[0]):
        if i == 0:
            tmp = wave_(data[i])
        else:
            tmp = np.vstack((tmp, wave_(data[i])))
    n_cols = tmp.shape[1]  # 获取处理后的列数
    new_labels = [f"wave_col_{i}" for i in range(n_cols)]  # 生成新标签（如wave_col_0, wave_col_1...）
    return pd.DataFrame(tmp, columns=new_labels)


# 均值中心化
def CT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MeanScaler :(n_samples, n_features)
       """
    # for i in range(data.shape[0]):
    #     MEAN = np.mean(data[i])
    #     data[i] = data[i] - MEAN
    # return data
    if isinstance(data, pd.DataFrame):
        data = data.values
    result = data.copy()
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        result[i] = data[i] - MEAN
    return result

# 标准正态变换
def SNV(data):
    """
        :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after SNV :(n_samples, n_features)
    """
    # m = data.shape[0]
    # n = data.shape[1]
    # # 求标准差
    # data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # # 求平均值
    # data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # # SNV计算
    # data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    # return np.array(data_snv)
    if isinstance(data, pd.DataFrame):
        data = data.values
    return (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)

# 一阶导数
def D1(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after First derivative :(n_samples, n_features)
    """
    # n, p = data.shape
    # Di = np.ones((n, p - 1))
    # for i in range(n):
    #     Di[i] = np.diff(data[i])
    # return Di
    if isinstance(data, pd.DataFrame):
        data = data.values
    return np.diff(data, axis=1)

# 二阶导数
def D2(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after second derivative :(n_samples, n_features)
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


# 趋势校正(DT)
def DT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after DT :(n_samples, n_features)
    """
    lenth = data.shape[1]
    x = np.asarray(range(lenth), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)

    return out


# 多元散射校正
def MSC(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MSC :(n_samples, n_features)
    """
    # n, p = data.shape
    # msc = np.ones((n, p))
    #
    # for j in range(n):
    #     mean = np.mean(data, axis=0)
    #
    # # 线性拟合
    # for i in range(n):
    #     y = data[i, :]
    #     l = LinearRegression()
    #     l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
    #     k = l.coef_
    #     b = l.intercept_
    #     msc[i, :] = (y - b) / k
    # return msc
    data = data.values if isinstance(data, pd.DataFrame) else data
    mean = np.mean(data, axis=0)
    msc = np.zeros_like(data)
    lr = LinearRegression()

    for i in range(data.shape[0]):
        y = data[i, :]
        lr.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lr.coef_[0][0]
        b = lr.intercept_[0]
        msc[i, :] = (y - b) / k

    return pd.DataFrame(msc)

# 连续统去除
def CRP(data):
    if isinstance(data, pd.DataFrame):
        data = data.values
    x_ind = data.shape[0]
    data = pd.DataFrame(data)
    ar = []  # 创建空值存放结果
    for i in range(data.shape[0]):
        x = data.iloc[i, :]
        q_u = np.zeros(x.shape)
        # 在插值值前加上第一个值，这将强制模型对上包络模型使用相同的起点。
        u_x = [0, ]
        u_y = [x[0], ]
        # 波峰和波谷
        for k in range(1, len(x) - 1):
            if ((np.sign(x[k] - x[k - 1]) == 1) and (np.sign(x[k] - x[k + 1]) == 1)) or (
                    (np.sign(x[k] - x[k - 1]) == -1) and (np.sign(x[k] - x[k + 1])) == -1):
                u_x.append(k)
                u_y.append(x[k])
        u_x.append(len(x) - 1)  # 包络与原始数据切点x
        u_y.append(x[x_ind])  # 对应的值
        u_p = interp1d(u_x, u_y, kind='cubic', fill_value=0)  # 二阶拟合，三阶用cubic
        for k in range(0, len(x)):
            q_u[k] = u_p(k)
        cr = x / q_u
        ar.append(cr)
        out = np.array(ar)
    return out


# 倒数对数
def CL(data):
    data = 100 / data

    def SS(data):
        ss = StandardScaler()
        if isinstance(data, pd.DataFrame):
            sspd = pd.DataFrame(ss.fit_transform(data), columns=data.columns)
            return sspd
        else:
            return ss.fit_transform(data)

    data = SS(data)
    # data = data + 10  # 避免求log的变量出现负值
    data = np.where(data <= 0, 1e-10, data)
    data = np.log(data)
    return data