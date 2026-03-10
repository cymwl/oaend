from django.shortcuts import render

# Create your views here.
# views.py
import os
import pandas as pd
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json


def SPA(X, y, max_features=10):
    """SPA算法实现"""
    # 步骤1：初始化，选与y相关性最高的波段
    corr = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    init_idx = np.argmax(np.abs(corr))  # 选相关系数绝对值最大的波段
    selected_idxs = [init_idx]

    # 步骤2：迭代投影选择
    while len(selected_idxs) < max_features:
        # 已选特征的正交空间
        X_selected = X[:, selected_idxs]
        # 计算正交投影矩阵
        P = X_selected @ np.linalg.inv(X_selected.T @ X_selected) @ X_selected.T
        # 计算剩余波段的投影长度
        projection_lengths = []
        for i in range(X.shape[1]):
            if i in selected_idxs:
                projection_lengths.append(0)
                continue
            x_i = X[:, i].reshape(-1, 1)
            # 投影长度=||x_i - Px_i||（正交空间的向量长度）
            length = np.linalg.norm(x_i - P @ x_i)
            projection_lengths.append(length)
        # 选投影长度最大的波段
        next_idx = np.argmax(projection_lengths)
        selected_idxs.append(next_idx)

    return np.array(selected_idxs)


def load_real_data(es_file, y_file):
    """读取光谱数据和Cd含量数据"""
    # 读取光谱数据
    df_es = pd.read_excel(es_file, header=0, index_col=0)
    es_sample_count = len(df_es)
    sample_names = df_es.index.tolist()

    # 读取重金属含量数据
    df_y = pd.read_excel(y_file, header=0, usecols=[0])
    y_sample_count = len(df_y)

    # 校验样本数
    if es_sample_count != y_sample_count:
        raise ValueError(f"光谱样本数（{es_sample_count}）与重金属含量样本数（{y_sample_count}）不一致！")

    # 转换为numpy数组
    X = df_es.values
    Y = df_y.values
    wavelengths = df_es.columns.values.astype(float)

    return X, Y, wavelengths, sample_names


@csrf_exempt
def process_spa_data(request):
    """处理SPA算法请求"""
    if request.method != 'POST':
        return JsonResponse({'error': '只支持POST请求'}, status=400)

    try:
        # 获取上传的文件
        es_file = request.FILES.get('es_file')
        cd_file = request.FILES.get('cd_file')

        if not es_file or not cd_file:
            return JsonResponse({'error': '请上传两个文件'}, status=400)

        # 保存上传的文件到临时目录
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        es_path = os.path.join(temp_dir, es_file.name)
        cd_path = os.path.join(temp_dir, cd_file.name)

        with open(es_path, 'wb+') as destination:
            for chunk in es_file.chunks():
                destination.write(chunk)

        with open(cd_path, 'wb+') as destination:
            for chunk in cd_file.chunks():
                destination.write(chunk)

        # 读取数据
        X, Y, wavelengths, sample_names = load_real_data(es_path, cd_path)

        # 将Y从列向量转为1维数组
        y_1d = Y.ravel()

        # 数据归一化
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        X_norm = min_max_scaler.fit_transform(X)

        # 建模集测试集分割
        Xcal, Xval, ycal, yval = train_test_split(X_norm, y_1d, test_size=0.4, random_state=0)

        # 运行SPA算法
        var_sel = SPA(Xcal, ycal, max_features=10)

        # 获取筛选后的数据
        X_select = X[:, var_sel]
        selected_wavelengths = wavelengths[var_sel].tolist()

        # 准备返回数据
        result_data = {
            # 'sample_names': sample_names,
            # 'cd_values': y_1d.tolist(),
            # 'selected_wavelengths': selected_wavelengths,
            # 'selected_absorbance': X_select.tolist(),
            # 'var_sel': var_sel.tolist(),
            # 'original_wavelengths': wavelengths.tolist()[:100],  # 只返回前100个用于显示
            # 'original_absorbance': X[:20, :100].tolist()  # 返回前20个样本的前100个波长用于对比

            'sample_names': sample_names,
            'cd_values': y_1d.tolist(),
            'selected_wavelengths': selected_wavelengths,
            'selected_absorbance': X_select.tolist(),
            'var_sel': var_sel.tolist(),
            'original_wavelengths': wavelengths.tolist(),  # 返回全部波长
            'original_absorbance': X.tolist()  # 返回全部样本的全部波长（若数据量大，可限制前50个样本）
            # 可选：若数据量过大，改为 X[:50].tolist() → 限制前50个样本，避免前端渲染卡顿
        }

        # 保存结果到SPAData文件夹
        spa_data_dir = os.path.join(settings.BASE_DIR, 'SPAData')
        os.makedirs(spa_data_dir, exist_ok=True)

        save_path = os.path.join(spa_data_dir, 'SPA_筛选结果.xlsx')

        # 创建DataFrame保存结果
        df_es = pd.read_excel(es_path, header=0, index_col=0)
        absorbances = df_es.columns.values

        df_save = pd.DataFrame({
            "样本名": sample_names,
            "Cd含量": y_1d
        })

        selected_wavelengths_names = absorbances[var_sel]
        for i, wl in enumerate(selected_wavelengths_names):
            col_name = f"{wl:.3f}nm"
            df_save[col_name] = X_select[:, i]

        df_wavelengths = pd.DataFrame({
            "筛选波段索引": var_sel,
            "波长值(nm)": selected_wavelengths_names
        })

        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            df_save.to_excel(writer, sheet_name="SPA筛选后数据", index=False)
            df_wavelengths.to_excel(writer, sheet_name="筛选波段清单", index=False)

        # 清理临时文件
        os.remove(es_path)
        os.remove(cd_path)

        return JsonResponse({
            'success': True,
            'data': result_data,
            'save_path': save_path,
            'message': 'SPA处理完成，数据已保存'
        })

    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=500)