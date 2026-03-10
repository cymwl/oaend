from django.shortcuts import render

# Create your views here.
import os
import time
import numpy as np
import pandas as pd
import joblib
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error, mean_absolute_error


def load_data(es_excel_path, y_excel_path):
    """读取SPA筛选后的特征波段光谱数据和Cd含量数据"""
    # 读取SPA筛选后的特征波段光谱数据
    X_selected = pd.read_excel(es_excel_path, header=0, index_col=0)
    # 读取重金属Cd含量数据
    y = pd.read_excel(y_excel_path, header=0, usecols=[0])
    # 转为numpy数组（适配sklearn）
    X_selected = X_selected.values  # 特征数组：(样本数, 特征数)
    y = y.values.ravel()  # 标签转为一维数组（SVR要求标签是一维）
    return X_selected, y


@csrf_exempt
def process_svr_data(request):
    """处理SVR算法请求"""
    if request.method != 'POST':
        return JsonResponse({'error': '只支持POST请求'}, status=400)

    try:
        # 获取上传的文件
        es_file = request.FILES.get('es_file')
        cd_file = request.FILES.get('cd_file')

        # 获取参数
        kernel = request.POST.get('kernel', 'rbf')
        C = float(request.POST.get('C', 5.0))
        gamma = float(request.POST.get('gamma', 0.001))
        pca_components = int(request.POST.get('pca_components', 50))
        test_size = float(request.POST.get('test_size', 0.2))

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

        # 记录开始时间
        start_time = time.time()

        # 读取数据
        X, y = load_data(es_path, cd_path)

        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA降维
        pca = PCA(n_components=min(pca_components, X_scaled.shape[1]), random_state=42)
        X_scaled_pca = pca.fit_transform(X_scaled)

        # 划分训练集/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_pca, y, test_size=test_size, random_state=42
        )

        # SVR模型训练
        model_svr = SVR(kernel=kernel, C=C, gamma=gamma)
        model_svr.fit(X_train, y_train)

        # 预测
        y_pred = model_svr.predict(X_test)

        # 计算评估指标
        evs = explained_variance_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # 计算运行时间
        run_time = time.time() - start_time

        # 准备预测结果数据
        sample_indices = list(range(len(y_test)))

        # 排序以便更好的可视化
        sorted_indices = np.argsort(y_test)
        y_test_sorted = y_test[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        # 计算残差
        residuals = y_test - y_pred

        # 准备返回数据
        result_data = {
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_test_sorted': y_test_sorted.tolist(),
            'y_pred_sorted': y_pred_sorted.tolist(),
            'sample_indices': sample_indices,
            'sorted_indices': sorted_indices.tolist(),
            'residuals': residuals.tolist(),
            'metrics': {
                'evs': evs,
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'run_time': run_time
            },
            'pca_info': {
                'components': pca.n_components_,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_variance_ratio': sum(pca.explained_variance_ratio_)
            },
            'model_params': {
                'kernel': kernel,
                'C': C,
                'gamma': gamma,
                'pca_components': pca_components,
                'test_size': test_size
            },
            'data_info': {
                'samples_total': len(y),
                'features_original': X.shape[1],
                'features_after_pca': X_scaled_pca.shape[1],
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        }

        # 保存模型和相关数据到SvrData文件夹
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        svr_data_dir = os.path.join(settings.BASE_DIR, 'SvrData')
        os.makedirs(svr_data_dir, exist_ok=True)

        # 创建结果文件夹
        result_dir = os.path.join(svr_data_dir, f'svr_result_{timestamp}')
        os.makedirs(result_dir, exist_ok=True)

        # 保存模型
        model_path = os.path.join(result_dir, 'svr_model.pkl')
        scaler_path = os.path.join(result_dir, 'scaler.pkl')
        pca_path = os.path.join(result_dir, 'pca.pkl')

        joblib.dump(model_svr, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(pca, pca_path)

        # 保存预测结果到Excel
        result_excel_path = os.path.join(result_dir, 'svr_predictions.xlsx')

        # 创建结果DataFrame
        df_results = pd.DataFrame({
            '样本索引': sample_indices,
            '真实值': y_test,
            '预测值': y_pred,
            '残差': residuals,
            '绝对误差': np.abs(residuals)
        })

        # 保存评估指标
        df_metrics = pd.DataFrame({
            '指标': ['解释方差分数(EVS)', '决定系数(R²)', '均方误差(MSE)',
                     '均方根误差(RMSE)', '平均绝对误差(MAE)', '运行时间(秒)'],
            '值': [evs, r2, mse, rmse, mae, run_time]
        })

        # 保存PCA信息
        df_pca = pd.DataFrame({
            '主成分索引': list(range(1, pca.n_components_ + 1)),
            '解释方差比': pca.explained_variance_ratio_,
            '累计解释方差比': np.cumsum(pca.explained_variance_ratio_)
        })

        # 保存到Excel
        with pd.ExcelWriter(result_excel_path, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='预测结果', index=False)
            df_metrics.to_excel(writer, sheet_name='评估指标', index=False)
            df_pca.to_excel(writer, sheet_name='PCA信息', index=False)

        # 清理临时文件
        os.remove(es_path)
        os.remove(cd_path)

        return JsonResponse({
            'success': True,
            'data': result_data,
            'save_path': result_dir,
            'excel_path': result_excel_path,
            'message': f'SVR建模完成，测试集R²: {r2:.4f}'
        })

    except Exception as e:
        import traceback
        print(f"SVR处理错误: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=500)