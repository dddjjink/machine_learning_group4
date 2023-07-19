import numpy as np
from Model import Model


# 朴素贝叶斯分类器
class NB(Model):
    def __init__(self):
        self.prior = {}
        self.conditional_prob = {}

    def fit(self, X, y):
        # 计算先验概率
        unique_y = set(y)
        for label in unique_y:
            self.prior[label] = np.sum(y == label) / len(y)

        # 计算条件概率
        for feature in range(X.shape[1]):
            unique_feature = set(X[:, feature])
            for label in unique_y:
                key = (feature, label)
                self.conditional_prob[key] = {}
                for value in unique_feature:
                    self.conditional_prob[key][value] = np.sum(
                        (X[:, feature] == value) & (y == label)
                    ) / np.sum(y == label)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            posterior_prob = {}
            for label in self.prior:
                posterior_prob[label] = np.log(self.prior[label])
                for feature in range(X.shape[1]):
                    key = (feature, label)
                    if X[i, feature] in self.conditional_prob[key]:
                        posterior_prob[label] += np.log(self.conditional_prob[key][X[i, feature]])

            y_pred.append(max(posterior_prob, key=posterior_prob.get))

        return y_pred


# # 朴素贝叶斯分类器示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
# 
#     # 鸢尾花数据集
#     # 数据载入
#     iris = pd.read_csv('../data/Iris.csv')
#     # print(iris.head(10))
#     # 数据分割
#     x = iris.drop(['Species', 'Id'], axis=1).values
#     y = iris['Species'].values
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # print(x_train, x_test, y_train, y_test)
#     # 模型训练、预测
#     clf = NB()
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(test_predict)
# 
#     '''
#     对本例的红酒数据集不适用，LogisticRegression适用二分类问题
#     '''
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
#     clf = NB()
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
#     clf = NB()
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(test_predict)
