class KMeans(Model):
    def __init__(self):
        self.data=Dataset.Dataset
    #计算距离
    def dist(instance1,instance2):
        return math.sqrt(sum(power(instance1-instance2, 2)))
    #选取k个聚类中心
    def randCent(self, k):
        n = shape(self.data)[1]   #n为数据集的列数
        centroids = mat(zeros((k, n)))  # 创建k*n的零矩阵
        for j in range(n):  # 在每个维度内创建随机聚类中心
            minJ = min(self.data[:, j])   #每一维最小的数
            rangeJ = float(max(self.data[:, j]) - minJ)   #最大-最小
            centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
        return centroids
    #k均值聚类算法实现
    def k_means(self,k):
        m = shape(self.data)[0]   #m为数据集的行数，即数据集的样本数
        clusterAssment = mat(zeros((m, 2)))
        centroids = self.randCent(self.data,k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):  # 将每个数据点分配给最近的质心
                minDist = math.inf;
                minIndex = -1
                for j in range(k):
                    distJI = self.dist(centroids[j, :], self.data[i, :])
                    if distJI < minDist:
                        minDist = distJI;
                        minIndex = j
                if clusterAssment[i, 0] != minIndex: clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
            print(centroids)
            for cent in range(k):  # recalculate centroids
                ptsInClust = self.data[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
                centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
        return centroids, clusterAssment
