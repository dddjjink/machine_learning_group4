import numpy as np
from Model import Model


class KMeans(Model):
    def __init__(self, k=3, max_iterations=300):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    # 计算距离
    def dist(self, instance1, instance2):
        return np.sqrt(np.sum(np.power(np.array(instance1) - np.array(instance2), 2)))

    # k均值聚类算法实现
    def fit(self, data):
        random_indices = np.random.choice(len(data), self.k, replace=False)
        self.centroids = [data[i] for i in random_indices]  # 选取k个初始聚类中心
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

    def predict(self, data):
        predictions = []
        for point in data:
            distances = [self.dist(point, centroid) for centroid in self.centroids]
            closest_centroid_index = np.argmin(distances)
            predictions.append(closest_centroid_index)
        return predictions


# # KMeans示例用法
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
#     clf = KMeans()
#     clf.fit(x_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
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
#     clf = KMeans()
#     clf.fit(x_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
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
#     clf = KMeans()
#     clf.fit(x_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
