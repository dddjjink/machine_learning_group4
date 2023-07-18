import numpy as np


class DB:

    def __init__(self, X, labels, metric='euclidean'):
        self.X = X
        self.labels = labels
        self.metric = metric

    def __call__(self):
        return self.calculate()

    def calculate(self):
        num_clusters = len(set(self.labels))
        cluster_centers = self._calculate_cluster_centers(self.X, self.labels)
        cluster_distances = self._calculate_cluster_distances(cluster_centers)
        cluster_similarities = self._calculate_cluster_similarities(cluster_distances)

        db_index = np.mean(np.max(cluster_similarities, axis=1))

        print(db_index)

    def _calculate_cluster_centers(self, X, labels):
        cluster_centers = []
        for i in range(len(set(labels))):
            points = X[labels == i]
            if len(points) == 0:
                continue
            cluster_center = np.mean(points, axis=0)
            cluster_centers.append(cluster_center)
        return np.array(cluster_centers)

    def _calculate_cluster_distances(self,cluster_centers):
        num_clusters = len(cluster_centers)
        cluster_distances = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                cluster_distances[i, j] = distance
                cluster_distances[j, i] = distance
        return cluster_distances

    def _calculate_cluster_similarities(self, cluster_distances):
        num_clusters = cluster_distances.shape[0]
        cluster_similarities = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(num_clusters):
                if i != j:
                    if cluster_distances[i, i] == 0 or cluster_distances[j, j] == 0:
                        similarity = 0
                    else:
                        similarity = (cluster_distances[i, j] + cluster_distances[j, i]) / cluster_distances[i, i]
                    cluster_similarities[i, j] = similarity
        return cluster_similarities



# # DB指数示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.cluster import KMeans
# 
#     # 鸢尾花数据集
#     # 数据载入
#     iris = pd.read_csv('../data/Iris.csv')
#     # print(iris.head(10))
#     # 数据分割
#     x = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
#     y = iris['Species'].values
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # print(x_train, x_test, y_train, y_test)
#     # 模型训练、预测
#     model = KMeans(n_clusters=3, random_state=1)
#     model.fit(x_train, y_train)
#     labels = model.labels_
#     # # 模型评估
#     db_train = DB(iris, labels)
#     db_test = DB(iris, labels)
#     db_train()
#     db_test()
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
#     model = KMeans(n_clusters=3, random_state=1)
#     model.fit(x_train, y_train)
#     labels = model.labels_
#     # # 模型评估
#     db_train = DB(wine, labels)
#     db_test = DB(wine, labels)
#     db_train()
#     db_test()
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
#     model = KMeans(n_clusters=3, random_state=1)
#     model.fit(x_train, y_train)
#     labels = model.labels_
#     # # 模型评估
#     db_train = DB(heart, labels)
#     db_test = DB(heart, labels)
#     db_train()
#     db_test()
