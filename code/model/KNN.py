import Model
import numpy as np
import operator
import math

# KNN模型，K最近邻算法->回归、分类
class KNN(Model):
    def __init__(self, k):
        self.k = k
        self.train_data = None

    #计算每个点之间的距离
    def distancecount(self,instance1,instance2):
        distance=0
        for i in range(len(instance1)):
            distance+=pow((instance1[i]-instance2[i]),2)
        return math.sqrt(distance)
    #获取k个邻居
    def kneighbors(self,test_data):
        distance1=[]
        length=len(test_data)-1
        for i in range(len(self.train_data)):
            dis = self.distancecount(test_data, self.train_data[i])
            distance1.append((self.train_data[i],dis))
        distance1.sort(key=operator.itemgetter(1))
        kneighbors=[]
        for i in range(self.k):
            kneighbors.append(distance1[i][0])
            return kneighbors
    #获取最多类别的类
    def most(self,neighbors):
        class1={}
        for neighbor in neighbors:
            most=int(neighbor[-1])
            if most in class1:
                class1[most]+=1
            else:
                class1[most]=1
        sortclass=sorted(class1.items(),key=operator.itemgetter(1),reverse=True)
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
