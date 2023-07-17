import evaluation
import numpy as np

class DB(Evaluation):
    def __init__(self, data,labels):
        self.data=data
        self.labels=labels

    def cal_distance(self, v1, v2): #    计算欧氏距离
        sum = 0
        for i in range(len(v1)):
            sum += (v1[i] - v2[i]) ** 2
        return sum ** 0.5

    def cal_center(self, cluster):
        # 计算聚类中心点
        center = [sum(i) / len(i) for i in zip(*cluster)]
        return center

    def calculate_similarity_matrix(self):# 计算数据集中所有点之间的相似度矩阵
        n = len(self.data)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self.cal_distance(self.data[i], self.data[j])
        return similarity_matrix

    def cal_disimilarity(self, cluster):# 计算簇内所有点之间的平均距离
        distances = [self.cal_distance(cluster[i], cluster[j])
                     for i in range(len(cluster)) for j in range(i+1, len(cluster))]
        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
        else:
            avg_distance = 0
        return avg_distance

    def cal_cluster_distance(self, cluster1, cluster2):# 计算两个聚类之间的距离
        distances = [self.cal_distance(point1, point2) for point1 in cluster1 for point2 in cluster2]
        min_distance = min(distances)
        return min_distance

    def DB_index(self):
        similarity_matrix = self.calculate_similarity_matrix()
        n = len(similarity_matrix)
        db_index = 0
        for i in range(n):
            in_distances = []#簇内距
            for j in range(n):
                if i != j:
                    in_distances.append(self.cal_disimilarity([self.data[i], self.data[j]]))
            avg_in_distance = sum(in_distances) / len(in_distances)
            inter_cluster_distances = []#簇间距
            for j in range(n):
                if i != j:
                    inter_cluster_distances.append(
                        self.cal_cluster_distance([self.data[i]], [self.data[j]]))
            max_inter_cluster_distance = max(inter_cluster_distances)
            db_index += (avg_in_distance / max_inter_cluster_distance)
        db_index /= n
        return db_index
