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
import traceback
import time
import random

@csrf_exempt
def cars(request):
    if request.method == 'POST':
        try:
            # 获取上传的文件
            spectral_file = request.FILES.get('spectral_file')
            heavy_metal_file = request.FILES.get('heavy_metal_file')

            if not spectral_file or not heavy_metal_file:
                return JsonResponse({'error': '请上传两个文件'}, status=400)

            # 读取原始数据（保留原始DataFrame，用于后续提取筛选后数据）
            spectral_df = pd.read_excel(spectral_file, header=0, index_col=None)
            heavy_metal_df = pd.read_excel(heavy_metal_file, header=0, index_col=None)

            # 数据格式校验
            if spectral_df.shape[1] < 2:
                return JsonResponse({'error': '光谱文件至少需要2列（第一列样本名+后续列光谱数据）'}, status=400)
            if heavy_metal_df.shape[1] < 1:
                return JsonResponse({'error': '重金属文件至少需要1列（重金属含量）'}, status=400)

            # 提取X（光谱数据）、Y（重金属含量），保留原始光谱的列名（用于筛选后数据的列名）
            X = spectral_df.iloc[:, 1:].values  # 光谱数据（跳过第一列样本名）
            Y = heavy_metal_df.iloc[:, 0].values  # 重金属含量
            spectral_feature_names = spectral_df.columns[1:].tolist()  # 原始光谱特征列名（波段名）

            # 校验样本数一致
            if len(X) != len(Y):
                return JsonResponse({'error': f'样本数不匹配！光谱数据{len(X)}个样本，重金属数据{len(Y)}个样本'}, status=400)

            # 读取前端传递的算法参数（默认值）
            N = int(request.POST.get('N', 100))
            f = int(request.POST.get('f', 8))
            cv = int(request.POST.get('cv', 5))

            # 数据标准化函数
            def standardize_data(data):
                # 处理缺失值（填充均值）
                data = np.nan_to_num(data, nan=np.nanmean(data, axis=0))
                # 计算标准差时避免为0
                std = np.std(data, axis=0)
                std[std < 1e-6] = 1e-6  # 用极小值替换0标准差
                return (data - np.mean(data, axis=0)) / std

            # 标准化数据
            x_std = standardize_data(X)
            y_std = standardize_data(Y.reshape(-1, 1)).flatten()

            # 运行CARS算法（返回最优特征的原始索引）
            opt_wave, rmse_m = cars_algorithm(x_std, y_std, N=N, f=f, cv=cv)

            # -------------------------- 优化1：保存筛选后的完整光谱数据 --------------------------
            # 1. 构建保存目录（媒体文件夹下的CarsData，自动创建）
            cars_data_dir = os.path.join(settings.MEDIA_ROOT, 'CarsData')
            os.makedirs(cars_data_dir, exist_ok=True)

            # 2. 生成唯一标识符（时间戳+随机数），避免文件覆盖
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            random_str = str(random.randint(100, 999))  # 3位随机数，进一步避免冲突
            unique_id = f'{timestamp}_{random_str}'

            # 3. 提取筛选后的光谱数据（原始数据，非标准化）
            # opt_wave是筛选后的特征索引（对应X的列索引，即spectral_df.iloc[:,1:]的列索引）
            selected_spectral_data = spectral_df.iloc[:, [0] + [i+1 for i in opt_wave]]  # 保留第一列样本名 + 筛选后的特征列
            # 筛选后的特征列名（对应原始波段名）
            selected_feature_names = [spectral_feature_names[i] for i in opt_wave]
            # 更新列名（确保样本名列名不变，筛选后的特征列名正确）
            selected_spectral_cols = [spectral_df.columns[0]] + selected_feature_names
            selected_spectral_data.columns = selected_spectral_cols

            # 4. 保存筛选后的光谱数据（Excel文件）
            spectral_save_path = os.path.join(cars_data_dir, f'selected_spectral_data_{unique_id}.xlsx')
            selected_spectral_data.to_excel(spectral_save_path, index=False)

            # 5. 保存特征索引和信息（补充筛选后的特征名，方便对应）
            feature_info_df = pd.DataFrame({
                'selected_feature_indices': opt_wave,  # 原始光谱数据的列索引（跳过样本名后）
                'selected_feature_names': selected_feature_names,  # 原始波段名
                'feature_display_name': [f'波段 {i + 1}' for i in opt_wave]  # 前端展示名
            })
            feature_info_save_path = os.path.join(cars_data_dir, f'selected_feature_info_{unique_id}.xlsx')
            feature_info_df.to_excel(feature_info_save_path, index=False)

            # 6. 保存算法参数和性能指标（方便复现）
            metrics_df = pd.DataFrame({
                '参数名': ['迭代次数N', '最大主成分数f', '交叉验证折数cv', '最小RMSE', '筛选后特征数', '生成时间'],
                '参数值': [N, f, cv, round(rmse_m, 4), len(opt_wave), timestamp]
            })
            metrics_save_path = os.path.join(cars_data_dir, f'algorithm_metrics_{unique_id}.xlsx')
            metrics_df.to_excel(metrics_save_path, index=False)

            # -------------------------- 准备返回给前端的数据 --------------------------
            response_data = {
                'selected_features': opt_wave.tolist(),
                'selected_feature_names': selected_feature_names,  # 返回原始波段名
                'display_feature_names': [f'波段 {i + 1}' for i in opt_wave],
                'rmse': float(rmse_m),
                'feature_count': len(opt_wave),
                'save_files': {
                    'spectral_data': os.path.basename(spectral_save_path),
                    'feature_info': os.path.basename(feature_info_save_path),
                    'metrics': os.path.basename(metrics_save_path)
                }
            }

            return JsonResponse({
                'success': True,
                'message': 'CARS特征选择完成，已保存筛选后的光谱数据',
                'data': response_data
            })

        except Exception as e:
            # 打印完整错误堆栈
            print("="*50)
            print("CARS算法执行错误详情：")
            traceback.print_exc()
            print("="*50)
            return JsonResponse({
                'success': False,
                'error': f'处理过程中出错: {str(e)}',
                'detail': traceback.format_exc()
            }, status=500)

    return JsonResponse({'error': '只支持POST请求'}, status=400)

def cars_algorithm(X, y, N=100, f=20, cv=10):
    """
    修复后的CARS算法：基于PLS系数排序筛选特征
    X: 光谱矩阵 nxm（n样本数，m特征数）
    y: 因变量（n维数组）
    N: 迭代次数
    f: 最大主成分数
    cv: 交叉验证数量
    return:
    OptWave: 最优特征的原始索引（对应输入X的列索引）
    RMSE_M: 交叉验证最小均方根误差
    """
    p = 0.8  # 蒙特卡洛重采样比例
    m, n = X.shape  # m样本数，n初始特征数
    u = np.power((n / 2), (1 / (N - 1)))  # 指数衰减因子基数
    k = (1 / (N - 1)) * np.log(n / 2)      # 衰减系数
    cal_num = int(np.round(m * p))         # 每次重采样的训练样本数
    current_features = np.arange(n)        # 当前保留的特征索引（初始为所有特征）
    x = copy.deepcopy(X)
    WaveData = []  # 记录每次迭代的特征掩码（0=剔除，1=保留）
    RMSECV = []    # 记录每次迭代的RMSE

    for i in range(1, N + 1):
        # 1. 计算当前迭代的特征保留数量（指数衰减）
        r_i = u * np.exp(-k * i)
        wave_num = int(np.round(r_i * len(current_features)))
        wave_num = max(5, wave_num)  # 最小保留5个特征，避免建模失败

        # 2. 蒙特卡洛重采样
        cal_index = np.random.choice(np.arange(m), size=cal_num, replace=False)
        xcal = x[cal_index, :]
        ycal = y[cal_index]

        # 3. PLS建模并计算特征系数
        f_current = wave_num if wave_num < f else f
        pls = PLSRegression(n_components=f_current)
        pls.fit(xcal, ycal)
        beta = pls.coef_.flatten()  # 特征系数（每个特征对应一个系数）

        # 4. 按系数绝对值排序，筛选最优特征
        coef_abs = np.abs(beta)
        sorted_indices = np.argsort(-coef_abs)  # 降序排序
        selected_coef_indices = sorted_indices[:wave_num]
        current_features = current_features[selected_coef_indices]  # 更新当前特征索引（映射到原始索引）

        # 5. 更新特征矩阵
        x = x[:, selected_coef_indices]

        # 6. 记录特征掩码（用于后续找到最优子集）
        feature_mask = np.zeros(n, dtype=int)
        feature_mask[current_features] = 1
        WaveData.append(feature_mask)

        # 7. 交叉验证计算RMSE
        rmsecv, rindex = pc_cross_validation(xcal, ycal, f_current, cv)
        current_rmse = cross_validation(xcal, ycal, rindex + 1, cv)
        RMSECV.append(current_rmse)

    # 8. 确定最优特征子集（RMSE最小时对应的掩码）
    RMSE_M = np.min(RMSECV)
    MinIndex = np.argmin(RMSECV)
    optimal_mask = WaveData[MinIndex]
    OptWave = np.where(optimal_mask == 1)[0]

    print(f'CARS算法完成：最优特征数={len(OptWave)}, 最小RMSE={RMSE_M:.4f}')
    return OptWave, RMSE_M

def pc_cross_validation(X, y, pc, cv):
    """主成分交叉验证：选择最优主成分数"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)  # 增加shuffle，结果更稳定
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
        RMSECV.append(np.mean(RMSE))
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex

def cross_validation(X, y, pc, cv):
    """交叉验证：计算指定主成分数的RMSE"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    return np.mean(RMSE)