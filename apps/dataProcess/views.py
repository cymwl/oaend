from django.shortcuts import render
# views.py
# Create your views here.
import os
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .pre import WAVE, CT, SNV, D1, D2, DT, MSC, CL, CRP


@csrf_exempt
def preprocess(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '仅支持POST请求'})

    # 获取文件和预处理方法
    file = request.FILES.get('file')
    method = request.POST.get('method')

    if not file or not method:
        return JsonResponse({'status': 'error', 'message': '缺少文件或处理方法'})

    try:
        # 读取Excel文件
        df = pd.read_excel(file,header=0, index_col=0)

        # 应用预处理方法
        method_map = {
            'WAVE': WAVE,
            'CT': CT,
            'SNV': SNV,
            'D1': D1,
            'D2': D2,
            'DT': DT,
            'MSC': MSC,
            'CL': CL,
            'CRP': CRP
        }

        if method not in method_map:
            return JsonResponse({'status': 'error', 'message': '未知的预处理方法'})

        processor = method_map[method]
        processed_data = processor(df)

        # 确保处理后的数据是二维数组格式
        if isinstance(processed_data, pd.DataFrame):
            processed_data = processed_data.values
        elif not isinstance(processed_data, np.ndarray):
            processed_data = np.array(processed_data)

        # 保存处理结果到本地
        output_path = f"processed_{method}_{file.name}"
        pd.DataFrame(processed_data).to_excel(output_path, index=False)

        # 返回处理结果（只返回部分数据避免过大）
        return JsonResponse({
            'status': 'success',
            'message': '处理完成',
            'data': processed_data.tolist()[:100]  # 限制返回数据量
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'处理过程中发生错误: {str(e)}'
        })

