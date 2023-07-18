# import numpy as np
# 
# import Model
# import math
# import random
# from numpy import power, shape, mat, zeros, nonzero, mean
# 
# 
class KMeans(Model):
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    #计算距离
    def dist(self,instance1,instance2):
        return np.sqrt(np.sum(np.power(np.array(instance1) - np.array(instance2), 2)))

    #k均值聚类算法实现
    def k_means(self,data):
        random_indices = np.random.choice(len(data), self.k, replace=False)
        self.centroids = [data[i] for i in random_indices]# 选取k个初始聚类中心
        for _ in range(self.max_iterations):
            clusters = [[] for _ in range(self.k)]
            # 分配每个样本到最近的聚类中心
            for point in data:
                distances = [self.dist(point, centroid) for centroid in self.centroids]
                closest_centroid_index = np.argmin(distances)
                clusters[closest_centroid_index].append(point)
            # 更新聚类中心
            for i in range(self.k):
                if clusters[i]:
                    self.centroids[i] = np.mean(clusters[i], axis=0)

    def fit(self, data):
        self.k_means(data)

    def predict(self, data):
        predictions = []
        for point in data:
            distances = [self.dist(point, centroid) for centroid in self.centroids]
            closest_centroid_index = np.argmin(distances)
            predictions.append(closest_centroid_index)

        return predictions


import numpy as np
from Model import Model


class K_Means(Model):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index

# # KMeans示例用法
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
#     # clf = K_Means()
#     # clf.fit(x_train)
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
#     # clf = K_Means()
#     # clf.fit(x_train)
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
#     clf = K_Means()
#     clf.fit(x_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     print(y_test)
#     print(test_predict)
