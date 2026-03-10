from django.core.serializers.json import DjangoJSONEncoder
from django.shortcuts import render

# Create your views here.
# views.py
import os
import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

# 新增：自定义JSON编码器处理NumPy类型
class NumpyJSONEncoder(DjangoJSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_real_data(es_excel_path, y_excel_path):
    """
    读取光谱数据和Cd含量数据
    """
    # 读取光谱数据
    df_es = pd.read_excel(es_excel_path, header=0, index_col=0)
    es_sample_count = len(df_es)
    sample_names = df_es.index.tolist()

    # 读取Cd含量数据
    df_y = pd.read_excel(y_excel_path, header=0, usecols=[0])
    y_sample_count = len(df_y)

    # 校验样本数
    if es_sample_count != y_sample_count:
        raise ValueError(f"光谱样本数（{es_sample_count}）与Cd含量样本数（{y_sample_count}）不一致！")

    # 转换为numpy数组
    X = df_es.values
    Y = df_y.values
    wavelengths = df_es.columns.values.astype(float)

    return X, Y, wavelengths, sample_names


def SiPLS(X, Y, wavelengths, interval_length=10):
    """
    SiPLS算法实现
    """
    n_intervals = X.shape[1] // interval_length
    scores = []
    interval_wls_list = []

    for i in range(n_intervals):
        start_idx = i * interval_length
        end_idx = start_idx + interval_length
        X_interval = X[:, start_idx:end_idx]
        # PLS建模
        pls = PLSRegression(n_components=min(2, X_interval.shape[1]))
        mse = -np.mean(cross_val_score(pls, X_interval, Y.ravel(), cv=5, scoring='neg_mean_squared_error'))
        scores.append(mse)
        interval_wls_list.append(wavelengths[start_idx:end_idx])

    # 找到最优区间
    best_interval_idx = np.argmin(scores)
    best_interval_wls = interval_wls_list[best_interval_idx]
    best_start = best_interval_idx * interval_length
    best_end = best_start + interval_length
    X_best = X[:, best_start:best_end]

    return scores, wavelengths[
                   :n_intervals * interval_length:interval_length], best_interval_idx, best_interval_wls, X_best


@csrf_exempt
def process_sipls_data(request):
    """处理SiPLS算法请求"""
    if request.method != 'POST':
        return JsonResponse({'error': '只支持POST请求'}, status=400)

    try:
        # 获取上传的文件
        es_file = request.FILES.get('es_file')
        cd_file = request.FILES.get('cd_file')
        interval_length = int(request.POST.get('interval_length', 10))

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

        # 运行SiPLS算法
        scores, interval_starts, best_idx, best_wls, X_best = SiPLS(
            X, Y, wavelengths, interval_length
        )

        # 准备返回数据
        result_data = {
            'sample_names': sample_names,
            'cd_values': Y.ravel().tolist(),
            'wavelengths': wavelengths.tolist(),
            'original_spectra': X[:20, :].tolist(),  # 只返回前20个样本用于展示
            'scores': scores,
            'interval_starts': interval_starts.tolist(),
            'best_interval_index': best_idx,
            'best_wavelengths': best_wls.tolist(),
            'best_spectra': X_best.tolist(),
            'interval_length': interval_length
        }

        # 保存结果到SIPLSData文件夹
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sipls_data_dir = os.path.join(settings.BASE_DIR, 'SIPLSData')
        os.makedirs(sipls_data_dir, exist_ok=True)

        # 保存Excel数据
        save_excel_path = os.path.join(sipls_data_dir, f'sipls_result_{timestamp}.xlsx')

        df_es = pd.read_excel(es_path, header=0, index_col=0)

        # 创建结果DataFrame
        df_result = pd.DataFrame({
            'Sample_Name': sample_names,
            'Cd_Content': Y.ravel()
        })

        for i, wl in enumerate(best_wls):
            df_result[f'WL_{wl:.1f}nm'] = X_best[:, i]

        # 保存区间得分信息
        df_scores = pd.DataFrame({
            'Interval_Start_Wavelength': interval_starts,
            'MSE_Score': scores,
            'Is_Best': [i == best_idx for i in range(len(scores))]
        })

        # 保存到Excel（两个sheet）
        with pd.ExcelWriter(save_excel_path, engine='openpyxl') as writer:
            df_result.to_excel(writer, sheet_name='SiPLS筛选后数据', index=False)
            df_scores.to_excel(writer, sheet_name='区间得分信息', index=False)

        # 生成图表（可选）
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt

            plot_path = os.path.join(sipls_data_dir, f'sipls_plot_{timestamp}.png')

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(wavelengths, X.T, color='grey', alpha=0.3)
            plt.axvspan(best_wls.min(), best_wls.max(), color='red', alpha=0.2)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Absorbance')
            plt.title(f'Original Spectra - Best Interval: {best_wls.min():.1f}~{best_wls.max():.1f}nm')
            plt.grid(alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.plot(interval_starts, scores, marker='o', color='darkblue', label='Interval MSE')
            plt.scatter(interval_starts[best_idx], scores[best_idx], color='red', s=100,
                        label=f'Best MSE: {scores[best_idx]:.4f}')
            plt.xlabel('Interval Start Wavelength (nm)')
            plt.ylabel('MSE (Lower = Better)')
            plt.title(f'SiPLS Scores (Interval Length = {interval_length})')
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()

            plot_url = f'/media/sipls_plots/sipls_plot_{timestamp}.png'

        except Exception as e:
            print(f"图表生成失败: {e}")
            plot_url = None

        # 清理临时文件
        os.remove(es_path)
        os.remove(cd_path)

        return JsonResponse({
            'success': True,
            'data': result_data,
            'save_path': save_excel_path,
            'plot_url': plot_url,
            'message': f'SiPLS处理完成，最优区间: {best_wls.min():.1f}~{best_wls.max():.1f}nm'
        }, encoder=NumpyJSONEncoder)


    except Exception as e:
        import traceback
        print(f"SiPLS处理错误: {str(e)}")
        print(traceback.format_exc())
        # 修改：异常返回时也使用自定义编码器
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=500, encoder=NumpyJSONEncoder)