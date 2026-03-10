from django.shortcuts import render

# Create your views here.
# views.py (XGBoost部分)
import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


@csrf_exempt
def xgboost_process(request):
    """处理XGBoost建模请求"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are supported', 'success': False}, status=400)

    try:
        print("=== Starting XGBoost processing ===")
        print(f"Request method: {request.method}")
        print(f"Request FILES keys: {list(request.FILES.keys())}")

        # 检查是否有文件上传
        if 'spectral_file' not in request.FILES:
            return JsonResponse({
                'error': 'spectral_file is required',
                'success': False
            }, status=400)

        if 'heavy_metal_file' not in request.FILES:
            return JsonResponse({
                'error': 'heavy_metal_file is required',
                'success': False
            }, status=400)

        # 获取上传的文件
        spectral_file = request.FILES['spectral_file']
        heavy_metal_file = request.FILES['heavy_metal_file']

        print(f"Spectral file name: {spectral_file.name}, size: {spectral_file.size}")
        print(f"Heavy metal file name: {heavy_metal_file.name}, size: {heavy_metal_file.size}")

        # 获取前端传递的XGBoost参数
        default_params = {
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'random_state': 42
        }

        # 尝试从POST数据中获取参数
        xgb_params = default_params.copy()
        if request.POST:
            try:
                # 如果前端使用FormData传递参数
                if 'xgb_params' in request.POST:
                    frontend_params = json.loads(request.POST['xgb_params'])
                    xgb_params.update(frontend_params)
                else:
                    # 尝试从其他可能的字段获取
                    for param in ['max_depth', 'learning_rate', 'n_estimators',
                                  'reg_alpha', 'reg_lambda', 'subsample',
                                  'colsample_bytree', 'gamma']:
                        if param in request.POST:
                            value = request.POST[param]
                            if param in ['max_depth', 'n_estimators', 'random_state']:
                                xgb_params[param] = int(value)
                            else:
                                xgb_params[param] = float(value)
            except Exception as e:
                print(f"Warning: Failed to parse XGBoost params: {e}, using defaults")

        print(f"XGBoost parameters: {xgb_params}")

        # 保存文件到临时目录
        fs = FileSystemStorage()
        timestamp = int(time.time())
        spectral_path = fs.save(f'temp_spectral_{timestamp}.xlsx', spectral_file)
        heavy_metal_path = fs.save(f'temp_heavy_metal_{timestamp}.xlsx', heavy_metal_file)

        spectral_full_path = os.path.join(fs.location, spectral_path)
        heavy_metal_full_path = os.path.join(fs.location, heavy_metal_path)

        # 读取数据
        def load_data(es_excel_path, y_excel_path):
            try:
                # 尝试读取光谱数据
                try:
                    X_selected = pd.read_excel(es_excel_path, header=0, index_col=0)
                except:
                    X_selected = pd.read_excel(es_excel_path, header=0)

                # 尝试读取重金属数据
                try:
                    y = pd.read_excel(y_excel_path, header=0, usecols=[0])
                except:
                    y = pd.read_excel(y_excel_path, header=None, usecols=[0])

                # 转换为numpy数组
                X_selected = X_selected.values
                y = y.values.ravel()  # 展平为一维数组

                return X_selected, y
            except Exception as e:
                raise ValueError(f"Error reading Excel files: {str(e)}")

        X, y = load_data(spectral_full_path, heavy_metal_full_path)
        print(f"Data loaded: feature shape {X.shape}, label shape {y.shape}")

        # 检查数据是否为空
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Data is empty. Please check your Excel files.")

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 划分训练/测试集（8:2）
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # ===================== XGBoost建模 =====================
        print("\n===== Starting XGBoost modeling =====")

        # 构建XGBoost参数
        params = {
            'objective': 'reg:squarederror',
            'max_depth': int(xgb_params.get('max_depth', 4)),
            'learning_rate': float(xgb_params.get('learning_rate', 0.05)),
            'n_estimators': int(xgb_params.get('n_estimators', 300)),
            'reg_alpha': float(xgb_params.get('reg_alpha', 0.5)),
            'reg_lambda': float(xgb_params.get('reg_lambda', 1.0)),
            'subsample': float(xgb_params.get('subsample', 0.8)),
            'colsample_bytree': float(xgb_params.get('colsample_bytree', 0.8)),
            'gamma': float(xgb_params.get('gamma', 0.1)),
            'random_state': int(xgb_params.get('random_state', 42))
        }

        # 创建模型
        xgb_model = xgb.XGBRegressor(**params)

        # 5折交叉验证
        print("Running 5-fold cross validation...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="r2")
        cv_r2_mean = round(float(cv_scores.mean()), 4)
        cv_r2_std = round(float(cv_scores.std()), 4)

        print(f"5-fold CV R2: mean={cv_r2_mean}, std={cv_r2_std}")

        # 训练模型
        print("Training XGBoost model...")
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # ===================== 预测与评估 =====================
        print("\n===== Making predictions and evaluating =====")

        # 训练集预测
        y_pred_train = xgb_model.predict(X_train)
        # 测试集预测
        y_pred_test = xgb_model.predict(X_test)

        # 计算评估指标
        train_r2 = round(r2_score(y_train, y_pred_train), 4)
        train_rmse = round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4)
        train_mae = round(mean_absolute_error(y_train, y_pred_train), 4)
        train_evs = round(explained_variance_score(y_train, y_pred_train), 4)

        test_r2 = round(r2_score(y_test, y_pred_test), 4)
        test_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4)
        test_mae = round(mean_absolute_error(y_test, y_pred_test), 4)
        test_evs = round(explained_variance_score(y_test, y_pred_test), 4)

        # 特征重要性分析
        importance = xgb_model.feature_importances_

        # 提取波长信息（假设第一行是波长）
        try:
            wave_data = pd.read_excel(spectral_full_path, header=None)
            if wave_data.shape[0] > 0:
                wave_selected = wave_data.iloc[0, 1:].values.ravel()[:len(importance)]
                # 清理波长数据
                wave_selected_cleaned = []
                for w in wave_selected:
                    if isinstance(w, str):
                        w_clean = w.replace('nm', '').strip()
                        try:
                            wave_selected_cleaned.append(float(w_clean))
                        except:
                            wave_selected_cleaned.append(0.0)
                    else:
                        wave_selected_cleaned.append(float(w))
                wave_selected = wave_selected_cleaned
            else:
                wave_selected = list(range(1, len(importance) + 1))
        except:
            wave_selected = list(range(1, len(importance) + 1))

        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            "特征索引": range(len(importance)),
            "波长(nm)": wave_selected,
            "重要性得分": importance
        }).sort_values(by="重要性得分", ascending=False)

        # 取前20个最重要的特征
        top_features = importance_df.head(20).to_dict('records')

        # ===================== 保存模型和结果 =====================
        # 创建保存目录
        save_dir = os.path.join(settings.BASE_DIR, 'XgboostData')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存模型
        model_path = os.path.join(save_dir, f'xgboost_model_{timestamp}.pkl')
        result_path = os.path.join(save_dir, f'xgboost_results_{timestamp}.npy')

        # 保存模型和scaler
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': xgb_model,
                'scaler': scaler,
                'params': params,
                'xgb_params': xgb_params,
                'timestamp': timestamp,
                'feature_importance': importance_df.to_dict(),
                'data_info': {
                    'feature_shape': X.shape,
                    'train_samples': len(y_train),
                    'test_samples': len(y_test)
                }
            }, f)

        # 保存结果数据
        np.save(result_path, {
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'y_train': y_train,
            'y_pred_train': y_pred_train,
            'metrics': {
                'train': {
                    'r2': train_r2,
                    'rmse': train_rmse,
                    'mae': train_mae,
                    'evs': train_evs
                },
                'test': {
                    'r2': test_r2,
                    'rmse': test_rmse,
                    'mae': test_mae,
                    'evs': test_evs
                },
                'cv': {
                    'r2_mean': cv_r2_mean,
                    'r2_std': cv_r2_std
                }
            },
            'feature_importance': importance_df.to_dict(),
            'top_features': top_features,
            'params': params,
            'xgb_params': xgb_params,
            'timestamp': timestamp
        })

        # ===================== 准备返回数据 =====================
        # 只取前50个测试样本用于展示
        sample_count = min(50, len(y_test))

        # 确保数据是可序列化的
        importance_serializable = []
        for _, row in importance_df.head(10).iterrows():
            importance_serializable.append({
                'feature_index': int(row['特征索引']),
                'wavelength': float(row['波长(nm)']),
                'importance': float(row['重要性得分'])
            })

        response_data = {
            'success': True,
            'message': 'XGBoost modeling successful',
            'data': {
                'indices': list(range(1, sample_count + 1)),
                'true_values': [float(val) for val in y_test[:sample_count]],
                'predicted_values': [float(val) for val in y_pred_test[:sample_count]],
                'metrics': {
                    'train': {
                        'r2': train_r2,
                        'rmse': train_rmse,
                        'mae': train_mae,
                        'evs': train_evs
                    },
                    'test': {
                        'r2': test_r2,
                        'rmse': test_rmse,
                        'mae': test_mae,
                        'evs': test_evs
                    },
                    'cv': {
                        'r2_mean': cv_r2_mean,
                        'r2_std': cv_r2_std
                    }
                },
                'feature_importance': importance_serializable,
                'top_features': top_features[:10],  # 只返回前10个
                'model_info': {
                    'name': f'xgboost_model_{timestamp}.pkl',
                    'timestamp': timestamp,
                    'params': params,
                    'data_info': {
                        'total_samples': len(y),
                        'train_samples': len(y_train),
                        'test_samples': len(y_test),
                        'feature_count': X.shape[1]
                    }
                }
            }
        }

        # 清理临时文件
        try:
            os.remove(spectral_full_path)
            os.remove(heavy_metal_full_path)
            print(f"Cleaned up temporary files")
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {str(e)}")

        print("=== XGBoost processing completed successfully ===")
        return JsonResponse(response_data)

    except Exception as e:
        print(f"Error during XGBoost processing: {str(e)}")
        import traceback
        traceback.print_exc()

        return JsonResponse({
            'error': f'Processing failed: {str(e)}',
            'success': False
        }, status=500)