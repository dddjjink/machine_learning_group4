import Model
import numpy as np

# KNN模型，K最近邻算法->回归、分类
class KNN(Model):
    #计算每个点之间的距离
    def distancecount(instance1,instance2,length):
        distance=0
        for i in range(length):
            distance+=pow((instance1[i]-instance2[i]),2)
        return math.sqrt(distance)
    #获取k个邻居
    def kneighbors(train_data,test_data,k):
        distance1=[]
        length=len(test_data)-1
        for i in range(len(train_data)):
            dis = train_data.distancecount(test_data, train_data[i], length)
            distance1.append(train_data[i],dis)
        distance1.sort(key=operator.itemgetter(1))
        kneightbors=[]
        for i in range(k):
            kneightbors.append(distance1[i][0])
            return kneightbors
    #获取最多类别的类
    def most(kneightbors):
        class1=[]
        for i in range(len(kneightbors)):
            most=kneightbors[i][-1]
            if most in class1:
                class1[most]+=1
            else:
                class1[most]=1
        sortclass=sorted(class1.items(),key=operator.itemgetter(1),reverse=True)
        return sortclass
