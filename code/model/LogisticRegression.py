import numpy as np
from .Model import Model
from .PCA import PCA


# 逻辑回归模型，二分类
class LogisticRegression(Model):
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        if X.ndim > 4:
            pca = PCA(2)
            X = pca.transform(X)
            y = pca.transform(y)
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 梯度下降算法
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # 更新权重和偏置项
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


# # LogisticRegression示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#
#     # '''
#     # 对本例的鸢尾花数据集不适用，LogisticRegression适用二分类问题
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
#     # clf = LogisticRegression()
#     # clf.fit(x_train, y_train)
#     # train_predict = clf.predict(x_train)
#     # test_predict = clf.predict(x_test)
#     # print(test_predict)
#
#     # '''
#     # 对本例的红酒数据集不适用，LogisticRegression适用二分类问题
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
#     # clf = LogisticRegression()
#     # clf.fit(x_train, y_train)
#     # train_predict = clf.predict(x_train)
#     # test_predict = clf.predict(x_test)
#     # print(train_predict)
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
#     clf = LogisticRegression()
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(test_predict)
