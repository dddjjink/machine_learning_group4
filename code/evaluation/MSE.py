from Evaluation import Evaluation
import numpy as np


# 均方误差
class MSE(Evaluation):
    def __init__(self, y_test, y_pred):
        super().__init__(y_test, y_pred)

    def __call__(self):
        self.loss()

    def loss(self):
        print(0.5 * np.sum((self.y_pred - self.y_true) ** 2, axis=-1))


# # MSE示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import LogisticRegression
#
#     # '''
#     # 对本例目前使用的鸢尾花数据集不适用，若要使用该数据集，需要先对target数据进行处理（str->int）
#     # '''
#     # # 鸢尾花数据集
#     # # 数据载入
#     # iris = pd.read_csv('../data/Iris.csv')
#     # # print(iris.head(10))
#     # # 数据分割
#     # x = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
#     # y = iris['Species'].values
#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # # print(x_train, x_test, y_train, y_test)
#     # # 模型训练、预测
#     # clf = LogisticRegression(random_state=0, solver='lbfgs')
#     # clf.fit(x_train, y_train)
#     # train_predict = clf.predict(x_train)
#     # test_predict = clf.predict(x_test)
#     # # 模型评估
#     # mse_train = MSE(y_train, train_predict)
#     # mse_test = MSE(y_test, test_predict)
#     # mse_train()
#     # mse_test()
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
#     clf = LogisticRegression(random_state=0, solver='lbfgs')
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     # 模型评估
#     mse_train = MSE(y_train, train_predict)
#     mse_test = MSE(y_test, test_predict)
#     mse_train()
#     mse_test()
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
#     clf = LogisticRegression(random_state=0, solver='lbfgs')
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     # 模型评估
#     mse_train = MSE(y_train, train_predict)
#     mse_test = MSE(y_test, test_predict)
#     mse_train()
#     mse_test()
