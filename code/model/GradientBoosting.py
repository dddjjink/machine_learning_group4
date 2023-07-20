import numpy as np
from .Model import Model


# 梯度增强算法，分类
class GBDT(Model):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        # 初始化预测值
        predictions = np.zeros(len(y))

        for _ in range(self.n_estimators):
            # 计算残差
            residual = y - predictions
            # 训练一个决策树
            tree = self._build_tree(X, residual, depth=0)
            self.trees.append(tree)
            # 更新预测值
            predictions += self.learning_rate * self.predict_tree(X, tree)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += self.learning_rate * self.predict_tree(X, tree)
        return predictions

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1:
            return np.mean(y)

        best_split_feature, best_split_value, best_gain = None, None, -np.inf
        best_left_indices, best_right_indices = None, None

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices = np.where(X[:, feature] <= value)[0]
                right_indices = np.where(X[:, feature] > value)[0]
                gain = self.calculate_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_split_feature = feature
                    best_split_value = value
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_gain == 0:
            return np.mean(y)

        left_tree = self._build_tree(X[best_left_indices], y[best_left_indices], depth + 1)
        right_tree = self._build_tree(X[best_right_indices], y[best_right_indices], depth + 1)
        return {'feature': best_split_feature, 'value': best_split_value, 'left': left_tree, 'right': right_tree}

    def predict_tree(self, X, tree):
        if isinstance(tree, np.float64):
            return np.full(len(X), tree)
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            if X[i, tree['feature']] <= tree['value']:
                predictions[i] = self.predict_tree(X[i].reshape(1, -1), tree['left'])
            else:
                predictions[i] = self.predict_tree(X[i].reshape(1, -1), tree['right'])
        return predictions

    def calculate_gain(self, parent, left, right):
        return self.calculate_mse(parent) - (len(left) / len(parent)) * self.calculate_mse(left) \
            - (len(right) / len(parent)) * self.calculate_mse(right)

    def calculate_mse(self, labels):
        if len(labels) == 0:
            return 0
        mean_value = np.mean(labels)
        mse = np.mean((labels - mean_value) ** 2)
        return mse


# # GDBT示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
# 
#     # '''
#     # 对本例目前使用的鸢尾花数据集不适用，若要使用该数据集，需要先对target数据进行处理（str->int）
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
#     # clf = GBDT()
#     # clf.fit(x_train, y_train)
#     # train_predict = clf.predict(x_train)
#     # test_predict = clf.predict(x_test)
#     # print(test_predict)
# 
#     # 红酒数据集
#     # 数据载入
#     wine = pd.read_csv('../data/WineQT.csv')
#     # print(wine.head(10))
#     # 数据分割
#     x = wine.drop(['quality', 'Id'], axis=1).values
#     y = wine['quality'].values
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # print(x_train, x_test, y_train, y_test)
#     # 模型训练、预测
#     clf = GBDT()
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(train_predict)
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
#     clf = GBDT()
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(test_predict)
