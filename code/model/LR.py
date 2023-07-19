import numpy as np
from Model import Model
from PCA import PCA


class LinearRegression(Model):
    def __init__(self):
        """初始化Linear Regression模型"""
        # 系数向量（θ1,θ2,.....θn）
        self.coef_ = None
        # 截距 (θ0)
        self.interception_ = None
        # θ向量
        self._theta = None

    def fit(self, X_train, y_train):
        if X_train.ndim > 4:
            pca = PCA(2)
            X_train = pca.transform(X_train)
            y_train = pca.transform(y_train)
        """根据训练数据集X_train，y_train 训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # np.ones((len(X_train), 1)) 构造一个和X_train 同样行数的，只有一列的全是1的矩阵
        # np.hstack 拼接矩阵
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # X_b.T 获取矩阵的转置
        # np.linalg.inv() 获取矩阵的逆
        # dot() 矩阵点乘
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.coef_ is not None and self.interception_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)


# # LinearRegression示例用法
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
#     # clf = LinearRegression()
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
#     clf = LinearRegression()
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
#     clf = LinearRegression()
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(test_predict)
