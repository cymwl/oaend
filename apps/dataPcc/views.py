from django.shortcuts import render

# Create your views here.
# views.py
import os
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import uuid
from .pcc import PCC  # 导入PCC函数


@csrf_exempt
def pcc(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '仅支持POST请求'})

    # 获取文件和参数
    spectrum_file = request.FILES.get('spectrum_file')
    heavy_metal_file = request.FILES.get('heavy_metal_file')
    feature_count = request.POST.get('feature_count')

    if not spectrum_file or not heavy_metal_file or not feature_count:
        return JsonResponse({'status': 'error', 'message': '缺少必要参数'})

    try:
        # 转换特征数量为整数
        feature_count = feature_count.strip()
        feature_count = int(feature_count)
        if feature_count < 1:
            return JsonResponse({'status': 'error', 'message': '特征数量必须大于0'})

        # 读取光谱数据
        spectrum_df = pd.read_excel(spectrum_file, header=0, index_col=0)

        # 读取重金属含量数据
        heavy_metal_df = pd.read_excel(heavy_metal_file,header=None).reset_index(drop=True)

        if isinstance(heavy_metal_df, pd.DataFrame):
            if heavy_metal_df.shape[1] != 1:
                raise ValueError("目标变量 Y 必须是单列数据")
            Y = heavy_metal_df.iloc[:, 0]  # 取第一列作为 Series
            Y = Y.drop(0).reset_index(drop=True)

        # 检查数据维度
        # if spectrum_df.shape[0] != heavy_metal_df.shape[0]:
        #     return JsonResponse({
        #         'status': 'error',
        #         'message': '光谱数据和重金属数据样本数量不一致'
        #     })

        # 执行PCC分析
        # 假设重金属数据只有一列，取第一列作为Y
        # Y = heavy_metal_df.iloc[:, 0]

        print(Y)
        X = spectrum_df

        print(X)

        # 调用PCC函数
        X_selected, _ = PCC(Y, X, feature_count)

        print(X_selected)
        # 设置输出目录
        output_dir = os.path.join(settings.BASE_DIR, 'TzData')
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

        # 生成唯一的文件名
        unique_id = uuid.uuid4().hex[:6]
        output_filename = f"pcc_features_{feature_count}_{unique_id}.xlsx"
        output_path = os.path.join(output_dir, output_filename)

        # 保存特征光谱数据
        X_selected.to_excel(output_path, index=False)

        # 返回处理结果（只返回特征光谱数据）
        return JsonResponse({
            'status': 'success',
            'message': '分析完成',
            'filename': output_filename,
            'data': X_selected.values.tolist()  # 返回全部数据
        })

    except ValueError as e:
        print(f"ValueError:{str(e)}")
        return JsonResponse({'status': 'error', 'message': '特征数量必须是整数'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'message': f'分析过程中发生错误: {str(e)}'
        })

