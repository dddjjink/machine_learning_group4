import Model
import math
import random
from numpy import power, shape, mat, zeros, nonzero, mean

class KMeans(Model):
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None

    #计算距离
    def dist(self,instance1,instance2):
        return math.sqrt(sum(power(instance1-instance2, 2)))
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
