from Model import Model
import operator
import math


# KNN模型，K最近邻算法->回归、分类
class KNN(Model):
    def __init__(self, k=3):
        self.k = k
        self.train_data = None

    # 计算每个点之间的距离
    def distancecount(self, instance1, instance2):
        distance = 0
        for i in range(len(instance1)):
            distance += pow((instance1[i] - instance2[i]), 2)
        return math.sqrt(distance)

    # 获取k个邻居
    def kneighbors(self, test_data):
        distance1 = []
        length = len(test_data) - 1
        for i in range(len(self.train_data)):
            dis = self.distancecount(test_data, self.train_data[i])
            distance1.append((self.train_data[i], dis))
        distance1.sort(key=operator.itemgetter(1))
        kneighbors = []
        for i in range(self.k):
            kneighbors.append(distance1[i][0])
            return kneighbors

    # 获取最多类别的类
    def most(self, neighbors):
        class1 = {}
        for neighbor in neighbors:
            most = int(neighbor[-1])
            if most in class1:
                class1[most] += 1
            else:
                class1[most] = 1
        sortclass = sorted(class1.items(), key=operator.itemgetter(1), reverse=True)
        return sortclass[0][0]

    # 拟合训练数据
    def fit(self, train_data):
        self.train_data = train_data

    # 预测测试数据的类别
    def predict(self, test_data):
        predictions = []
        for data in test_data:
            neighbors = self.kneighbors(data)
            predicted_class = self.most(neighbors)
            predictions.append(predicted_class)
        return predictions


# # KNN示例用法
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
#     clf = KNN()
#     clf.fit(x_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(test_predict)
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
#     clf = KNN()
#     clf.fit(x_train)
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
#     clf = KNN()
#     clf.fit(x_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(test_predict)
