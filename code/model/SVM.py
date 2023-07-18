from Model import Model
import numpy as np


# 支持向量机模型，二分类
class SVM(Model):
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel

    # 参数初始化
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        self.alpha = np.ones(self.m)
        self.computer_product_matrix()  # 为了加快训练速度创建一个内积矩阵
        # 松弛变量
        self.C = 1.0
        # 将Ei保存在一个列表里
        self.create_e()

    # KKT条件判断
    def judge_kkt(self, i):
        y_g = self.function_g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # 计算内积矩阵，如果数据量较大，可以使用系数矩阵
    def computer_product_matrix(self):
        self.product_matrix = np.zeros((self.m, self.m)).astype(np.float)
        for i in range(self.m):
            for j in range(self.m):
                if self.product_matrix[i][j] == 0.0:
                    self.product_matrix[i][j] = self.product_matrix[j][i] = self.kernel(self.X[i], self.X[j])

    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return np.dot(x1, x2)
        elif self._kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** 2
        return 0

    # 将Ei保存在一个列表里
    def create_e(self):
        self.E = (np.dot((self.alpha * self.Y), self.product_matrix) + self.b) - self.Y

    # 预测函数g(x)
    def function_g(self, i):
        return self.b + np.dot((self.alpha * self.Y), self.product_matrix[i])

    # 选择变量
    def select_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self.judge_kkt(i):
                continue
            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = np.argmin(self.E)
            else:
                j = np.argmax(self.E)
            return i, j

    # 剪切
    def clip_alpha(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    # 训练函数，使用SMO算法，features, labels -> x_train, y_train
    def fit(self, features, labels):
        self.init_args(features, labels)
        # SMO算法训练
        for t in range(self.max_iter):
            i1, i2 = self.select_alpha()

            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
                self.X[i1], self.X[i2])
            if eta <= 0:
                # print('eta <= 0')
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            alpha2_new = self.clip_alpha(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.create_e()

    # 预测
    def predict(self, data):
        results = []
        for i in range(len(data)):
            r = self.b
            for j in range(self.m):
                r += self.alpha[j] * self.Y[j] * self.kernel(data[i], self.X[j])
            if r > 0:
                result = 1
            else:
                result = 0
            results.append(result)
        return results

# # SVM示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#
#     # '''
#     # 对本例的鸢尾花数据集不适用，SVM适用二分类问题
#     # '''
#     # # 鸢尾花数据集
#     # # 数据载入
#     # iris = pd.read_csv('../data/Iris.csv')
#     # # print(iris.head(10))
#     # # 数据分割
#     # x = iris.drop(['Species', 'Id'], axis=1).values
#     # y = iris['Species'].values
#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # # print(x_train, x_test, y_train, y_test)
#     # # 模型训练、预测
#     # clf = SVM()
#     # clf.fit(x_train, y_train)
#     # train_predict = clf.predict(x_train)
#     # test_predict = clf.predict(x_test)
#
#     # '''
#     # 对本例的红酒数据集不适用，SVM适用二分类问题
#     # '''
#     # # 红酒数据集
#     # # 数据载入
#     # wine = pd.read_csv('../data/WineQT.csv')
#     # # print(wine.head(10))
#     # # 数据分割
#     # x = wine.drop(['quality', 'Id'], axis=1).values
#     # y = wine['quality'].values
#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # # print(x_train, x_test, y_train, y_test)
#     # # 模型训练、预测
#     # clf = SVM()
#     # clf.fit(x_train, y_train)
#     # train_predict = clf.predict(x_train)
#     # test_predict = clf.predict(x_test)
#
#     # 心脏病数据集
#     # 数据载入
#     heart = pd.read_csv('../data/heart.csv')
#     # print(heart.head(10))
#     # 数据分割
#     x = heart.drop(['target'], axis=1).values
#     y = heart['target'].values
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # print(x_train, x_test, y_train, y_test)
#     # 模型训练、预测
#     clf = SVM()
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(y_test)
#     print(test_predict)

