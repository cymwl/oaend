import os
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import copy
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


@csrf_exempt
def cars(request):
    if request.method == 'POST':
        try:
            # 获取上传的文件
            spectral_file = request.FILES.get('spectral_file')
            heavy_metal_file = request.FILES.get('heavy_metal_file')

            if not spectral_file or not heavy_metal_file:
                return JsonResponse({'error': '请上传两个文件'}, status=400)

            # 读取数据
            spectral_df = pd.read_excel(spectral_file)
            # , header = 0, index_col = 0  .reset_index(drop=True)
            heavy_metal_df = pd.read_excel(heavy_metal_file)

            # 提取X和Y
            X = spectral_df.iloc[:, 1:].values  # 假设第一列是样本名
            Y = heavy_metal_df.iloc[:, 0].values  # 假设第一列是样本名，第二列是重金属含量

            # 数据标准化函数
            def standardize_data(data):
                return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

            # 标准化数据
            x_std = standardize_data(X)
            y_std = standardize_data(Y.reshape(-1, 1)).flatten()

            # 运行CARS算法
            opt_wave, rmse_m = cars_algorithm(x_std, y_std, N=50, f=10, cv=5)

            # 保存结果到CarsData文件夹
            cars_data_dir = os.path.join(settings.BASE_DIR, 'CarsData')
            os.makedirs(cars_data_dir, exist_ok=True)

            # 保存选择的特征数据
            selected_features_df = pd.DataFrame({
                'selected_feature_indices': opt_wave,
                'selected_feature_names': [f'Feature_{i}' for i in opt_wave]
            })
            selected_features_df.to_excel(os.path.join(cars_data_dir, 'selected_features.xlsx'), index=False)

            # 准备返回数据
            response_data = {
                'selected_features': opt_wave.tolist(),
                'rmse': float(rmse_m),
                'feature_names': [f'波段 {i + 1}' for i in opt_wave],
                'feature_count': len(opt_wave)
            }

            return JsonResponse({
                'success': True,
                'message': 'CARS特征选择完成',
                'data': response_data
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'处理过程中出错: {str(e)}'
            }, status=500)

    return JsonResponse({'error': '只支持POST请求'}, status=400)


def cars_algorithm(X, y, N=100, f=20, cv=10):
    """
    CARS算法特征选择
    X: 光谱矩阵 nxm
    y: 因变量
    N: 迭代次数
    f: 最大主成分数
    cv: 交叉验证数量
    return:
    OptWave: 最优变量索引
    RMSE_M: 交叉验证最小均方根误差
    """
    p = 0.8
    m, n = X.shape
    u = np.power((n / 2), (1 / (N - 1)))
    k = (1 / (N - 1)) * np.log(n / 2)
    cal_num = np.round(m * p)
    b2 = np.arange(n)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveData = []
    WaveNum = []
    RMSECV = []
    r = []

    for i in range(1, N + 1):
        r.append(u * np.exp(-1 * k * i))
        wave_num = int(np.round(r[i - 1] * n))
        WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice(np.arange(m), size=int(cal_num), replace=False)
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        ycal = y[cal_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1, -1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:
            WaveData = d
        else:
            WaveData = np.vstack((WaveData, d.reshape(1, -1)))
        if wave_num < f:
            f = wave_num
        pls = PLSRegression(n_components=f)
        pls.fit(xcal, ycal)
        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)
        coef = copy.deepcopy(beta)
        coeff = coef[b2, :].reshape(len(b2), -1)
        rmsecv, rindex = pc_cross_validation(xcal, ycal, f, cv)
        RMSECV.append(cross_validation(xcal, ycal, rindex + 1, cv))

    WAVE = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        WD = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
            else:
                WD[j] = wd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
    RMSE_M = np.min(RMSECV)
    print('CarsRMSE最小值:{:.4f}'.format(RMSE_M))
    MinIndex = np.argmin(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]

    return OptWave, RMSE_M


def pc_cross_validation(X, y, pc, cv):
    """
    主成分交叉验证
    X: 光谱矩阵 nxm
    y: 因变量
    pc: 最大主成分数
    cv: 交叉验证数量
    return:
    RMSECV: 各主成分数对应的RMSECV
    rindex: 最佳主成分数
    """
    kf = KFold(n_splits=cv)
    RMSECV = []
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex


def cross_validation(X, y, pc, cv):
    """
    交叉验证
    X: 光谱矩阵 nxm
    y: 因变量
    pc: 主成分数
    cv: 交叉验证数量
    return:
    RMSE_mean: 交叉验证的平均RMSE
    """
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean