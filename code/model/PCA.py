import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # 计算均值
        self.mean = np.mean(X, axis=0)
        # 中心化数据
        X = X - self.mean
        # 计算协方差矩阵
        cov = np.cov(X.T)
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # 对特征向量进行排序
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # 选择前n_components个特征向量
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # 中心化数据
        X = X - self.mean
        # 投影到主成分上
        return np.dot(X, self.components.T)
