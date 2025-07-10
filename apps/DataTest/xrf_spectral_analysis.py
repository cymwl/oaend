import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.stats import pearsonr
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class XRFDataProcessor:
    """XRF光谱数据处理与重金属含量反演类"""

    def __init__(self, wavelet='db4', decomp_level=5, corr_threshold=0.8,
                 n_woa_iterations=100, n_woa_agents=30):
        """初始化处理参数

        Args:
            wavelet: 小波变换类型
            decomp_level: 小波分解级别
            corr_threshold: 皮尔逊相关系数阈值，用于特征选择
            n_woa_iterations: 鲸鱼优化算法迭代次数
            n_woa_agents: 鲸鱼优化算法中代理数量
        """
        self.wavelet = wavelet
        self.decomp_level = decomp_level
        self.corr_threshold = corr_threshold
        self.n_woa_iterations = n_woa_iterations
        self.n_woa_agents = n_woa_agents
        self.selected_features = None
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self, spectrum_file, concentration_file, spectrum_id_col=0,
                  conc_id_col=0, conc_value_col=1):
        """加载光谱数据和浓度数据

        Args:
            spectrum_file: 光谱数据文件路径
            concentration_file: 浓度数据文件路径
            spectrum_id_col: 光谱数据中样本ID列的索引
            conc_id_col: 浓度数据中样本ID列的索引
            conc_value_col: 浓度数据中浓度值列的索引

        Returns:
            spectra: 光谱数据数组
            concentrations: 浓度数据数组
            wavelengths: 波长数组
            sample_ids: 样本ID数组
        """
        # 加载光谱数据
        spectrum_df = pd.read_csv(spectrum_file)
        wavelengths = spectrum_df.columns[1:].astype(float).values
        sample_ids = spectrum_df.iloc[:, spectrum_id_col].values
        spectra = spectrum_df.iloc[:, 1:].values

        # 加载浓度数据
        conc_df = pd.read_csv(concentration_file)
        conc_ids = conc_df.iloc[:, conc_id_col].values
        concentrations = conc_df.iloc[:, conc_value_col].values

        # 确保光谱数据和浓度数据样本顺序一致
        spectra_ordered = []
        concentrations_ordered = []

        for sid in sample_ids:
            if sid in conc_ids:
                idx = np.where(conc_ids == sid)[0][0]
                spectra_ordered.append(spectra[np.where(sample_ids == sid)[0][0]])
                concentrations_ordered.append(concentrations[idx])

        return np.array(spectra_ordered), np.array(concentrations_ordered), wavelengths, sample_ids

    def wavelet_denoising(self, spectra):
        """使用小波变换进行光谱去噪

        Args:
            spectra: 原始光谱数据

        Returns:
            denoised_spectra: 去噪后的光谱数据
        """
        denoised_spectra = []

        for spectrum in spectra:
            # 小波分解
            coeffs = pywt.wavedec(spectrum, self.wavelet, level=self.decomp_level)

            # 阈值处理（软阈值）
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(spectrum)))
            denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

            # 小波重构
            denoised_spectrum = pywt.waverec(denoised_coeffs, self.wavelet)

            # 确保重构后的光谱长度与原始光谱相同
            if len(denoised_spectrum) > len(spectrum):
                denoised_spectrum = denoised_spectrum[:len(spectrum)]
            elif len(denoised_spectrum) < len(spectrum):
                padding = np.zeros(len(spectrum) - len(denoised_spectrum))
                denoised_spectrum = np.concatenate([denoised_spectrum, padding])

            denoised_spectra.append(denoised_spectrum)

        return np.array(denoised_spectra)

    def spectral_transform(self, spectra, transform_type='snv'):
        """进行光谱变换

        Args:
            spectra: 光谱数据
            transform_type: 变换类型，可选'snv'(标准正态变量变换)、'msc'(多元散射校正)

        Returns:
            transformed_spectra: 变换后的光谱数据
        """
        if transform_type == 'snv':
            # 标准正态变量变换
            mean = np.mean(spectra, axis=1, keepdims=True)
            std = np.std(spectra, axis=1, keepdims=True)
            return (spectra - mean) / std

        elif transform_type == 'msc':
            # 多元散射校正
            # 计算平均光谱
            mean_spectrum = np.mean(spectra, axis=0)
            transformed_spectra = np.zeros_like(spectra)

            for i, spectrum in enumerate(spectra):
                # 线性回归拟合
                slope, intercept = np.polyfit(mean_spectrum, spectrum, 1)
                # 校正
                transformed_spectra[i] = (spectrum - intercept) / slope

            return transformed_spectra

        else:
            return spectra

    def feature_selection(self, spectra, concentrations, wavelengths):
        """使用皮尔逊相关系数进行特征选择

        Args:
            spectra: 光谱数据
            concentrations: 浓度数据
            wavelengths: 波长数组

        Returns:
            selected_spectra: 选择特征后的光谱数据
            selected_wavelengths: 选择的波长
        """
        n_features = spectra.shape[1]
        correlations = np.zeros(n_features)

        # 计算每个波长与浓度的皮尔逊相关系数
        for i in range(n_features):
            corr, _ = pearsonr(spectra[:, i], concentrations)
            correlations[i] = np.abs(corr)

        # 选择相关系数大于阈值的特征
        self.selected_features = np.where(correlations > self.corr_threshold)[0]

        if len(self.selected_features) == 0:
            # 如果没有特征超过阈值，选择相关系数最高的前10%特征
            n_top_features = max(1, int(n_features * 0.1))
            self.selected_features = np.argsort(correlations)[-n_top_features:]

        selected_spectra = spectra[:, self.selected_features]
        selected_wavelengths = wavelengths[self.selected_features]

        print(f"特征选择: 从 {n_features} 个特征中选择了 {len(self.selected_features)} 个特征")
        return selected_spectra, selected_wavelengths

    def woa_optimization(self, X_train, y_train, X_val, y_val):
        """使用鲸鱼优化算法优化模型参数

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签

        Returns:
            best_params: 最优参数
        """
        # 定义优化参数范围
        param_ranges = {
            'n_estimators': (50, 300),  # 决策树数量
            'max_depth': (3, 20),  # 树的最大深度
            'min_samples_split': (2, 10),  # 分裂内部节点所需的最小样本数
            'min_samples_leaf': (1, 5)  # 叶子节点所需的最小样本数
        }

        # 初始化鲸鱼种群
        n_params = len(param_ranges)
        whales = np.zeros((self.n_woa_agents, n_params))

        # 随机初始化鲸鱼位置（参数值）
        for i in range(self.n_woa_agents):
            for j, (param, (min_val, max_val)) in enumerate(param_ranges.items()):
                if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    whales[i, j] = np.random.randint(min_val, max_val + 1)
                else:
                    whales[i, j] = np.random.uniform(min_val, max_val)

        # 评估初始种群
        fitness = np.zeros(self.n_woa_agents)
        for i in range(self.n_woa_agents):
            params = self.decode_params(whales[i], param_ranges)
            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fitness[i] = r2_score(y_val, y_pred)

        # 记录最优解
        best_idx = np.argmax(fitness)
        best_whale = whales[best_idx].copy()
        best_fitness = fitness[best_idx]
        best_params = self.decode_params(best_whale, param_ranges)

        # 鲸鱼优化算法迭代
        a = 2  # 控制搜索范围的参数，从2线性减小到0

        for iteration in range(self.n_woa_iterations):
            a = 2 - iteration * (2 / self.n_woa_iterations)  # 更新a值

            for i in range(self.n_woa_agents):
                # 随机参数
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a  # 控制搜索方向
                C = 2 * r2  # 随机加权系数
                l = np.random.uniform(-1, 1)  # 螺旋更新参数
                p = np.random.rand()  # 决定搜索策略的概率

                for j in range(n_params):
                    if p < 0.5:
                        # 包围猎物或随机搜索
                        if abs(A) < 1:
                            # 包围猎物
                            D = abs(C * best_whale[j] - whales[i, j])
                            whales[i, j] = best_whale[j] - A * D
                        else:
                            # 随机搜索
                            random_idx = np.random.randint(0, self.n_woa_agents)
                            X_rand = whales[random_idx, j]
                            D = abs(C * X_rand - whales[i, j])
                            whales[i, j] = X_rand - A * D
                    else:
                        # 螺旋更新位置
                        D_prime = abs(best_whale[j] - whales[i, j])
                        whales[i, j] = D_prime * np.exp(l) * np.cos(2 * np.pi * l) + best_whale[j]

                    # 边界处理
                    min_val, max_val = list(param_ranges.values())[j]
                    if list(param_ranges.keys())[j] in ['n_estimators', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf']:
                        whales[i, j] = np.clip(int(whales[i, j]), min_val, max_val)
                    else:
                        whales[i, j] = np.clip(whales[i, j], min_val, max_val)

                # 评估新位置
                params = self.decode_params(whales[i], param_ranges)
                model = RandomForestRegressor(**params, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                new_fitness = r2_score(y_val, y_pred)

                # 更新最优解
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_whale = whales[i].copy()
                    best_params = self.decode_params(best_whale, param_ranges)

            # 打印迭代进度
            if (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration + 1}/{self.n_woa_iterations}, 最优R²: {best_fitness:.4f}")

        print(f"鲸鱼优化算法完成，最优参数: {best_params}")
        return best_params

    def decode_params(self, whale, param_ranges):
        """将鲸鱼位置解码为模型参数

        Args:
            whale: 鲸鱼位置向量
            param_ranges: 参数范围字典

        Returns:
            params: 模型参数字典
        """
        params = {}
        for i, (param, (min_val, max_val)) in enumerate(param_ranges.items()):
            if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                params[param] = int(whale[i])
            else:
                params[param] = whale[i]
        return params

    def train_model(self, spectra, concentrations, test_size=0.2, random_state=42):
        """训练预测模型

        Args:
            spectra: 光谱数据
            concentrations: 浓度数据
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            metrics: 模型评估指标
        """
        # 划分训练集和验证集
        X_train, X_test, y_train, y_test = train_test_split(
            spectra, concentrations, test_size=test_size, random_state=random_state
        )

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 进一步划分为训练集和验证集（用于WOA优化）
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.25, random_state=random_state
        )

        # 使用鲸鱼优化算法寻找最优参数
        print("正在使用鲸鱼优化算法寻找最优参数...")
        best_params = self.woa_optimization(X_train_split, y_train_split, X_val, y_val)

        # 使用最优参数训练最终模型
        print("正在使用最优参数训练最终模型...")
        self.model = RandomForestRegressor(**best_params, random_state=random_state)
        self.model.fit(X_train_scaled, y_train)

        # 在测试集上评估模型
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'rmse': rmse,
            'r2': r2
        }

        print(f"模型评估结果: RMSE={rmse:.4f}, R²={r2:.4f}")
        return metrics

    def predict(self, spectra):
        """使用训练好的模型进行预测

        Args:
            spectra: 光谱数据

        Returns:
            predictions: 预测的浓度值
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")

        # 特征选择
        if self.selected_features is not None:
            spectra = spectra[:, self.selected_features]

        # 数据标准化
        spectra_scaled = self.scaler.transform(spectra)

        # 预测
        predictions = self.model.predict(spectra_scaled)
        return predictions

    def visualize_results(self, spectra, concentrations, wavelengths, output_dir='results'):
        """可视化处理结果

        Args:
            spectra: 原始光谱数据
            concentrations: 浓度数据
            wavelengths: 波长数组
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 原始光谱可视化
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, spectra.T)
        plt.title('原始XRF光谱')
        plt.xlabel('波长 (nm)')
        plt.ylabel('强度')
        plt.savefig(os.path.join(output_dir, '原始光谱.png'))

        # 2. 去噪后光谱可视化
        denoised_spectra = self.wavelet_denoising(spectra)
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, denoised_spectra.T)
        plt.title('小波去噪后的XRF光谱')
        plt.xlabel('波长 (nm)')
        plt.ylabel('强度')
        plt.savefig(os.path.join(output_dir, '去噪后光谱.png'))

        # 3. 光谱变换后可视化
        transformed_spectra = self.spectral_transform(denoised_spectra)
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, transformed_spectra.T)
        plt.title('光谱变换后的XRF光谱')
        plt.xlabel('波长 (nm)')
        plt.ylabel('强度')
        plt.savefig(os.path.join(output_dir, '变换后光谱.png'))

        # 4. 特征选择结果可视化
        if self.selected_features is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(wavelengths, np.mean(transformed_spectra, axis=0), label='平均光谱')

            # 标记选择的特征
            for idx in self.selected_features:
                plt.axvline(x=wavelengths[idx], color='r', linestyle='--', alpha=0.3)

            plt.title('特征选择结果')
            plt.xlabel('波长 (nm)')
            plt.ylabel('强度')
            plt.legend()
            plt.savefig(os.path.join(output_dir, '特征选择结果.png'))

        # 5. 预测结果可视化
        if self.model is not None:
            predictions = self.predict(spectra)

            plt.figure(figsize=(10, 6))
            plt.scatter(concentrations, predictions, alpha=0.7)
            plt.plot([min(concentrations), max(concentrations)],
                     [min(concentrations), max(concentrations)], 'r--')
            plt.xlabel('实际浓度')
            plt.ylabel('预测浓度')
            plt.title('模型预测结果')
            plt.savefig(os.path.join(output_dir, '预测结果.png'))

    def save_model(self, model_path):
        """保存模型和处理参数

        Args:
            model_path: 模型保存路径
        """
        model_data = {
            'processor': self,
            'model': self.model,
            'selected_features': self.selected_features,
            'scaler': self.scaler
        }
        joblib.dump(model_data, model_path)
        print(f"模型已保存至: {model_path}")

    @staticmethod
    def load_model(model_path):
        """加载保存的模型

        Args:
            model_path: 模型保存路径

        Returns:
            processor: 加载的模型处理器
        """
        model_data = joblib.load(model_path)
        processor = model_data['processor']
        processor.model = model_data['model']
        processor.selected_features = model_data['selected_features']
        processor.scaler = model_data['scaler']
        return processor


def main():
    # 配置参数
    config = {
        'spectrum_file': 'xrf_spectra.csv',  # 光谱数据文件
        'concentration_file': 'heavy_metal_concentrations.csv',  # 重金属浓度文件
        'output_dir': 'xrf_results',  # 结果输出目录
        'wavelet': 'db4',  # 小波类型
        'decomp_level': 5,  # 小波分解级别
        'corr_threshold': 0.7,  # 皮尔逊相关系数阈值
        'n_woa_iterations': 50,  # 鲸鱼优化算法迭代次数
        'n_woa_agents': 20,  # 鲸鱼优化算法代理数量
        'spectrum_id_col': 0,  # 光谱数据中样本ID列的索引
        'conc_id_col': 0,  # 浓度数据中样本ID列的索引
        'conc_value_col': 1,  # 浓度数据中浓度值列的索引
        'spectral_transform_type': 'snv'  # 光谱变换类型
    }

    # 创建处理器实例
    processor = XRFDataProcessor(
        wavelet=config['wavelet'],
        decomp_level=config['decomp_level'],
        corr_threshold=config['corr_threshold'],
        n_woa_iterations=config['n_woa_iterations'],
        n_woa_agents=config['n_woa_agents']
    )

    # 加载数据
    print("正在加载数据...")
    spectra, concentrations, wavelengths, sample_ids = processor.load_data(
        config['spectrum_file'],
        config['concentration_file'],
        spectrum_id_col=config['spectrum_id_col'],
        conc_id_col=config['conc_id_col'],
        conc_value_col=config['conc_value_col']
    )

    # 小波去噪
    print("正在进行小波去噪...")
    denoised_spectra = processor.wavelet_denoising(spectra)

    # 光谱变换
    print("正在进行光谱变换...")
    transformed_spectra = processor.spectral_transform(
        denoised_spectra,
        transform_type=config['spectral_transform_type']
    )

    # 特征选择
    print("正在进行特征选择...")
    selected_spectra, selected_wavelengths = processor.feature_selection(
        transformed_spectra, concentrations, wavelengths
    )

    # 训练模型
    print("正在训练模型...")
    metrics = processor.train_model(selected_spectra, concentrations)

    # 可视化结果
    print("正在可视化结果...")
    processor.visualize_results(spectra, concentrations, wavelengths, config['output_dir'])

    # 保存预测结果
    predictions = processor.predict(spectra)
    results_df = pd.DataFrame({
        'Sample_ID': sample_ids,
        'Actual_Concentration': concentrations,
        'Predicted_Concentration': predictions,
        'Residual': concentrations - predictions
    })
    results_df.to_csv(os.path.join(config['output_dir'], 'prediction_results.csv'), index=False)
    print(f"预测结果已保存至: {os.path.join(config['output_dir'], 'prediction_results.csv')}")

    # 保存模型
    processor.save_model(os.path.join(config['output_dir'], 'xrf_model.pkl'))


if __name__ == "__main__":
    main()