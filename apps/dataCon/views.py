from django.shortcuts import render

# Create your views here.
import os
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
from datetime import datetime


@csrf_exempt
@api_view(['POST'])
def datacon(request):
    """融合XRF和PSR光谱数据"""
    try:
        # 获取上传的文件
        xrf_file = request.FILES.get('xrf_file')
        psr_file = request.FILES.get('psr_file')

        if not xrf_file or not psr_file:
            return Response({
                'success': False,
                'error': '请选择XRF和PSR两个文件'
            }, status=400)

        # 验证文件类型
        if not xrf_file.name.endswith('.xlsx') or not psr_file.name.endswith('.xlsx'):
            return Response({
                'success': False,
                'error': '只支持.xlsx格式文件'
            }, status=400)

        # 读取数据
        try:
            xrf_df = pd.read_excel(xrf_file, header=0, index_col=0)
            psr_df = pd.read_excel(psr_file, header=0, index_col=0)
            xrf_df = xrf_df.reset_index(drop=True)
            psr_df = psr_df.reset_index(drop=True)
            print(xrf_df)
            print(psr_df)
        except Exception as e:
            return Response({
                'success': False,
                'error': f'读取Excel文件失败: {str(e)}'
            }, status=400)

        # 数据融合
        try:
            Xall = pd.concat([xrf_df, psr_df], axis=1)

            print(Xall)
        except Exception as e:
            return Response({
                'success': False,
                'error': f'数据融合失败: {str(e)}'
            }, status=400)

        # 创建ConcatData文件夹
        concat_data_dir = os.path.join(settings.BASE_DIR, 'ConcatData')
        os.makedirs(concat_data_dir, exist_ok=True)

        # 保存融合后的数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'merged_spectral_data_{timestamp}.xlsx'
        output_path = os.path.join(concat_data_dir, output_filename)

        try:
            Xall.to_excel(output_path, index=True)
        except Exception as e:
            return Response({
                'success': False,
                'error': f'保存融合数据失败: {str(e)}'
            }, status=500)

        # 准备返回给前端的数据
        # 取前20行和前20列用于可视化（避免数据量过大）
        display_data = Xall.iloc[:20, :20].copy()

        display_columns = display_data.columns.tolist()  # 前端X轴标签
        display_index = display_data.index.tolist()  # 前端Y轴标签（样本名）
        # 转为列表格式（JSON可序列化）
        processed_data = display_data.values.tolist()

        # 6. 返回响应（关键：按前端预期结构返回success、data、columns、index、data_info）
        return JsonResponse({
            'success': True,  # 前端判断成功的核心字段
            'message': '处理完成',
            'data': processed_data,  # 融合后的数据（20×20）
            'columns': display_columns,  # 前端X轴标签（波长）
            'index': display_index,  # 前端Y轴标签（样本）
            # 补充数据维度信息（前端“融合结果”区域显示用）
            'data_info': {
                'xrf_shape': [xrf_df.shape[0], xrf_df.shape[1]],  # XRF数据维度
                'psr_shape': [psr_df.shape[0], psr_df.shape[1]],  # PSR数据维度
                'merged_shape': [Xall.shape[0], Xall.shape[1]]  # 融合后维度
            },
            'output_filename': output_filename  # 前端显示“输出文件”名
        })


    except Exception as e:
        return Response({
            'success': False,
            'error': f'处理过程中出错: {str(e)}'
        }, status=500)


# @api_view(['GET'])
# def get_merge_history(request):
#     """获取融合历史记录"""
#     try:
#         concat_data_dir = os.path.join(settings.BASE_DIR, 'ConcatData')
#         if not os.path.exists(concat_data_dir):
#             return Response({
#                 'success': True,
#                 'data': []
#             })
#
#         files = []
#         for filename in os.listdir(concat_data_dir):
#             if filename.endswith('.xlsx'):
#                 file_path = os.path.join(concat_data_dir, filename)
#                 file_stat = os.stat(file_path)
#                 files.append({
#                     'filename': filename,
#                     'size': file_stat.st_size,
#                     'created_time': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
#                 })
#
#         # 按创建时间倒序排列
#         files.sort(key=lambda x: x['created_time'], reverse=True)
#
#         return Response({
#             'success': True,
#             'data': files[:10]  # 返回最近10个文件
#         })
#
#     except Exception as e:
#         return Response({
#             'success': False,
#             'error': f'获取历史记录失败: {str(e)}'
#         }, status=500)