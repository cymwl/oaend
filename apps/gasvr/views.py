from django.shortcuts import render

# Create your views here.
import os
import json
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import time
import random
import copy
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import pickle
import warnings

warnings.filterwarnings('ignore')


# ===================== GA优化核心模块 =====================
def fitness_func(predictval, realval):
    """计算R²作为适应度值"""
    r2 = r2_score(realval, predictval)
    return r2


def svr_evaluate(vardim, params, bound, feature_train, target_train):
    """适配线性核SVR，增加异常处理"""
    try:
        # 提取参数并裁剪
        c = np.clip(params[0], bound[0, 0], bound[1, 0])
        e = np.clip(params[1], bound[0, 1], bound[1, 1])

        # 线性核SVR（小样本最优）
        svr_model = SVR(kernel='linear', C=c, epsilon=e)
        # 5折交叉验证，增加异常捕获
        cv_scores = cross_val_score(svr_model, feature_train, target_train, cv=5, scoring='r2', error_score=np.nan)
        cv_r2 = np.nanmean(cv_scores)  # 忽略nan值

        # 处理极端值（避免适应度为负无穷）
        if np.isnan(cv_r2) or cv_r2 < -10:
            cv_r2 = -10.0
        return cv_r2
    except:
        return -10.0  # 异常时返回保底值


class GAIndividual:
    def __init__(self, vardim, bound):
        self.vardim = vardim
        self.bound = bound
        self.chrom = np.zeros(vardim)
        self.fitness = 0.0

    def generate(self):
        """随机生成参数，增加参数合理性校验"""
        rnd = np.random.random(size=self.vardim)
        for i in range(self.vardim):
            self.chrom[i] = self.bound[0, i] + (self.bound[1, i] - self.bound[0, i]) * rnd[i]
            # 确保参数不为0
            if self.chrom[i] <= 0:
                self.chrom[i] = self.bound[0, i] + 1e-6

    def calculate_fitness(self, feature_train, target_train):
        self.fitness = svr_evaluate(self.vardim, self.chrom, self.bound, feature_train, target_train)


class GeneticAlgorithm:
    def __init__(self, pop_size, vardim, bound, max_gen, params, feature_train, target_train):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.vardim = vardim
        self.bound = bound
        self.params = params
        self.feature_train = feature_train
        self.target_train = target_train
        self.population = []
        self.fitness = np.zeros((self.pop_size, 1))
        self.trace = np.zeros((self.max_gen, 3))
        self.best_individual = None

    def initialize(self):
        """初始化种群，确保至少有一个有效个体"""
        for i in range(self.pop_size):
            ind = GAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)
        # 强制初始化一个最优初始参数（避免全负）
        self.population[0].chrom = np.array([1.0, 0.01])  # 经验最优初始值
        self.population[0].calculate_fitness(self.feature_train, self.target_train)

    def evaluate(self):
        """评估所有个体，处理异常值"""
        for i in range(self.pop_size):
            self.population[i].calculate_fitness(self.feature_train, self.target_train)
            self.fitness[i] = self.population[i].fitness
        # 将所有负适应度转换为相对值（解决轮盘赌失效）
        min_fitness = np.min(self.fitness)
        if min_fitness < 0:
            self.fitness = self.fitness - min_fitness + 0.001  # 平移到正数域

    def selection(self):
        """修复版轮盘赌选择：彻底避免索引越界"""
        new_pop = []
        total_fitness = np.sum(self.fitness)

        # 极端情况处理：所有适应度为0
        if total_fitness <= 1e-8:
            new_pop = copy.deepcopy(self.population)  # 直接复制原种群
        else:
            acc_fitness = np.cumsum(self.fitness / total_fitness)
            for i in range(self.pop_size):
                r = random.random()
                # 找到第一个大于等于r的索引，无则选最后一个
                mask = acc_fitness >= r
                if np.any(mask):
                    idx = np.where(mask)[0][0]
                else:
                    idx = self.pop_size - 1
                new_pop.append(copy.deepcopy(self.population[idx]))
        self.population = new_pop

    def crossover(self):
        """交叉操作：增加参数合理性校验"""
        new_pop = []
        for i in range(0, self.pop_size, 2):
            # 随机选两个不同父代
            idx1 = random.randint(0, self.pop_size - 1)
            idx2 = random.randint(0, self.pop_size - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.pop_size - 1)

            parent1 = copy.deepcopy(self.population[idx1])
            parent2 = copy.deepcopy(self.population[idx2])

            # 交叉概率判断
            if random.random() < self.params[0]:
                cross_pos = random.randint(1, self.vardim - 1)
                for j in range(cross_pos, self.vardim):
                    # 交叉计算
                    temp = parent1.chrom[j]
                    parent1.chrom[j] = self.params[2] * parent1.chrom[j] + (1 - self.params[2]) * parent2.chrom[j]
                    parent2.chrom[j] = self.params[2] * parent2.chrom[j] + (1 - self.params[2]) * temp
                    # 确保参数为正
                    parent1.chrom[j] = max(parent1.chrom[j], 1e-6)
                    parent2.chrom[j] = max(parent2.chrom[j], 1e-6)

            new_pop.append(parent1)
            if len(new_pop) < self.pop_size:
                new_pop.append(parent2)

        # 确保种群大小一致
        self.population = new_pop[:self.pop_size]

    def mutation(self):
        """变异操作：自适应步长+参数校验"""
        new_pop = []
        for i in range(self.pop_size):
            ind = copy.deepcopy(self.population[i])
            if random.random() < self.params[1]:
                mutate_pos = random.randint(0, self.vardim - 1)
                theta = random.random()
                # 自适应变异步长
                step = (1 - random.random() ** (1 - self.t / self.max_gen)) * 0.1

                if theta > 0.5:
                    new_val = ind.chrom[mutate_pos] * (1 - step)
                else:
                    new_val = ind.chrom[mutate_pos] * (1 + step)

                # 确保参数在范围内且为正
                new_val = np.clip(new_val, self.bound[0, mutate_pos], self.bound[1, mutate_pos])
                new_val = max(new_val, 1e-6)
                ind.chrom[mutate_pos] = new_val

            new_pop.append(ind)
        self.population = new_pop

    def solve(self):
        """主进化流程"""
        self.t = 0
        self.initialize()
        self.evaluate()

        # 初始化最优个体
        best_idx = np.argmax(self.fitness)
        self.best_individual = copy.deepcopy(self.population[best_idx])
        # 还原平移后的适应度值
        min_fitness = np.min(self.fitness) - 0.001
        real_best_fitness = self.best_individual.fitness + min_fitness - 0.001

        self.trace[self.t, 0] = real_best_fitness
        self.trace[self.t, 1] = np.mean(self.fitness) + min_fitness - 0.001
        self.trace[self.t, 2] = np.max(self.fitness) + min_fitness - 0.001

        # 修改为英文打印，避免编码问题
        print(f"Generation {self.t}: best CV-R2 = {self.trace[self.t, 0]:.4f}")

        # 迭代进化
        while self.t < self.max_gen - 1:
            self.t += 1
            self.selection()
            self.crossover()
            self.mutation()
            self.evaluate()

            # 更新最优个体
            current_best_idx = np.argmax(self.fitness)
            current_best = self.population[current_best_idx]
            real_current_fitness = current_best.fitness + min_fitness - 0.001

            if real_current_fitness > self.trace[self.t - 1, 0]:
                self.best_individual = copy.deepcopy(current_best)

            # 记录轨迹（还原真实R²）
            self.trace[self.t, 0] = self.best_individual.fitness + min_fitness - 0.001
            self.trace[self.t, 1] = np.mean(self.fitness) + min_fitness - 0.001
            self.trace[self.t, 2] = np.max(self.fitness) + min_fitness - 0.001

            print(f"Generation {self.t}: best CV-R2 = {self.trace[self.t, 0]:.4f}")

        # 输出结果
        print("\n===== GA optimization completed =====")
        real_final_fitness = self.best_individual.fitness + min_fitness - 0.001
        print(f"Best 5-fold CV-R2: {real_final_fitness:.4f}")
        print(f"Best parameters: C={self.best_individual.chrom[0]:.4f}, epsilon={self.best_individual.chrom[1]:.4f}")
        return self.best_individual.chrom, self.trace


@csrf_exempt
def ga_svr_process(request):
    """处理GA-SVR建模请求"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are supported'}, status=400)

    try:
        # 获取上传的文件
        spectral_file = request.FILES.get('spectral_file')
        heavy_metal_file = request.FILES.get('heavy_metal_file')

        if not spectral_file or not heavy_metal_file:
            return JsonResponse({'error': 'Please upload two Excel files'}, status=400)

        # 保存文件到临时目录
        fs = FileSystemStorage()
        spectral_path = fs.save(f'temp_spectral_{int(time.time())}.xlsx', spectral_file)
        heavy_metal_path = fs.save(f'temp_heavy_metal_{int(time.time())}.xlsx', heavy_metal_file)

        spectral_full_path = os.path.join(fs.location, spectral_path)
        heavy_metal_full_path = os.path.join(fs.location, heavy_metal_path)

        # 读取数据
        def load_data(es_excel_path, y_excel_path):
            X_all = pd.read_excel(es_excel_path, header=0, index_col=0).values
            y = pd.read_excel(y_excel_path, header=0, usecols=[0]).values.ravel()
            return X_all, y

        X, y = load_data(spectral_full_path, heavy_metal_full_path)
        print(f"Data loaded: feature shape {X.shape}, label shape {y.shape}")

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 划分训练/测试集（7:3）
        feature_train, feature_test, target_train, target_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, shuffle=True
        )

        # ===================== 运行GA优化 =====================
        # GA参数配置
        pop_size = 15
        vardim = 2
        bound = np.array([[0.001, 0.0001], [5.0, 0.05]])
        max_gen = 20
        ga_params = [0.6, 0.03, 0.5]

        print("\n===== Start GA optimization for linear kernel SVR =====")
        ga = GeneticAlgorithm(pop_size, vardim, bound, max_gen, ga_params, feature_train, target_train)
        best_params, trace = ga.solve()

        # ===================== 最终模型训练与评估 =====================
        print("\n===== Training final linear kernel SVR model =====")
        # 使用GA最优参数
        best_svr = SVR(kernel='linear', C=best_params[0], epsilon=best_params[1])
        best_svr.fit(feature_train, target_train)
        predict_results = best_svr.predict(feature_test)

        # 岭回归对比
        ridge = Ridge(alpha=best_params[0])
        ridge.fit(feature_train, target_train)
        ridge_pred = ridge.predict(feature_test)

        # 评估指标
        svr_r2 = round(r2_score(target_test, predict_results), 4)
        svr_evs = round(explained_variance_score(target_test, predict_results), 4)
        ridge_r2 = round(r2_score(target_test, ridge_pred), 4)

        # 创建保存目录
        save_dir = os.path.join(settings.BASE_DIR, 'GaSvrData')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存模型和结果
        timestamp = int(time.time())
        model_path = os.path.join(save_dir, f'ga_svr_model_{timestamp}.pkl')
        result_path = os.path.join(save_dir, f'ga_svr_results_{timestamp}.npy')

        # 保存模型和scaler
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best_svr,
                'scaler': scaler,
                'best_params': best_params,
                'feature_train_shape': feature_train.shape
            }, f)

        # 保存结果数据
        np.save(result_path, {
            'target_test': target_test,
            'predict_results': predict_results,
            'ridge_pred': ridge_pred,
            'trace': trace,
            'metrics': {
                'svr_r2': svr_r2,
                'svr_evs': svr_evs,
                'ridge_r2': ridge_r2
            }
        })

        # 准备返回给前端的数据
        # 只取前50个样本用于展示
        sample_count = min(50, len(target_test))

        response_data = {
            'success': True,
            'message': 'GA-SVR modeling successful',
            'data': {
                'indices': list(range(1, sample_count + 1)),
                'true_values': target_test[:sample_count].tolist(),
                'svr_predictions': predict_results[:sample_count].tolist(),
                'ridge_predictions': ridge_pred[:sample_count].tolist(),
                'metrics': {
                    'svr_r2': svr_r2,
                    'svr_evs': svr_evs,
                    'ridge_r2': ridge_r2
                },
                'trace': trace.tolist(),
                'best_params': best_params.tolist()
            }
        }
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(f"Return data: {response_data}")
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # 清理临时文件
        os.remove(spectral_full_path)
        os.remove(heavy_metal_full_path)

        return JsonResponse(response_data)

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return JsonResponse({
            'error': f'Processing failed: {str(e)}',
            'success': False
        }, status=500)


@csrf_exempt
def get_saved_models(request):
    """获取已保存的模型列表"""
    try:
        save_dir = os.path.join(settings.BASE_DIR, 'GaSvrData')
        if not os.path.exists(save_dir):
            return JsonResponse({'models': []})

        models = []
        for file in os.listdir(save_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(save_dir, file)
                create_time = os.path.getctime(file_path)
                models.append({
                    'name': file,
                    'path': file_path,
                    'create_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(create_time))
                })

        return JsonResponse({'models': models})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def load_model_results(request):
    """加载指定模型的结果"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are supported'}, status=400)

    try:
        data = json.loads(request.body)
        model_name = data.get('model_name')

        if not model_name:
            return JsonResponse({'error': 'Model name not specified'}, status=400)

        save_dir = os.path.join(settings.BASE_DIR, 'GaSvrData')
        result_file = model_name.replace('.pkl', '.npy')
        result_path = os.path.join(save_dir, result_file)

        if not os.path.exists(result_path):
            return JsonResponse({'error': 'Result file does not exist'}, status=404)

        # 加载结果数据
        loaded_data = np.load(result_path, allow_pickle=True).item()

        # 准备返回数据
        target_test = loaded_data['target_test']
        predict_results = loaded_data['predict_results']
        ridge_pred = loaded_data['ridge_pred']
        metrics = loaded_data['metrics']

        sample_count = min(50, len(target_test))

        response_data = {
            'success': True,
            'data': {
                'indices': list(range(1, sample_count + 1)),
                'true_values': target_test[:sample_count].tolist(),
                'svr_predictions': predict_results[:sample_count].tolist(),
                'ridge_predictions': ridge_pred[:sample_count].tolist(),
                'metrics': metrics
            }
        }

        return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)