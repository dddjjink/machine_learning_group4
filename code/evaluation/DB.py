import numpy as np


class DB:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __call__(self, *args, **kwargs):
        self.db_index()

    def cal_distance(self, v1, v2):  # 计算欧氏距离
        sum = 0
        for i in range(len(v1)):
            sum += (v1[i] - v2[i]) ** 2
        return sum ** 0.5

    def cal_center(self, cluster):
        # 计算聚类中心点
        center = [sum(i) / len(i) for i in zip(*cluster)]
        return center

    def calculate_similarity_matrix(self):  # 计算数据集中所有点之间的相似度矩阵
        n = len(self.data)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    '''
                    !!!!!!!!!!此处有误!!!!!!!!!!
                    '''                    
                    similarity_matrix[i][j] = self.cal_distance(self.data[i], self.data[j])
        return similarity_matrix

    def cal_disimilarity(self, cluster):  # 计算簇内所有点之间的平均距离
        distances = [self.cal_distance(cluster[i], cluster[j])
                     for i in range(len(cluster)) for j in range(i + 1, len(cluster))]
        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
        else:
            avg_distance = 0
        return avg_distance

    def cal_cluster_distance(self, cluster1, cluster2):  # 计算两个聚类之间的距离
        distances = [self.cal_distance(point1, point2) for point1 in cluster1 for point2 in cluster2]
        min_distance = min(distances)
        return min_distance

    def db_index(self):
        similarity_matrix = self.calculate_similarity_matrix()
        n = len(similarity_matrix)
        db_index = 0
        for i in range(n):
            in_distances = []  # 簇内距
            for j in range(n):
                if i != j:
                    in_distances.append(self.cal_disimilarity([self.data[i], self.data[j]]))
            avg_in_distance = sum(in_distances) / len(in_distances)
            inter_cluster_distances = []  # 簇间距
            for j in range(n):
                if i != j:
                    inter_cluster_distances.append(
                        self.cal_cluster_distance([self.data[i]], [self.data[j]]))
            max_inter_cluster_distance = max(inter_cluster_distances)
            db_index += (avg_in_distance / max_inter_cluster_distance)
        db_index /= n
        print(db_index)


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
