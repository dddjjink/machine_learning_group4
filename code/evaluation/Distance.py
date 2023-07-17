from Evaluation import Evaluation


# 距离公式
class Distance(Evaluation):
    def __init__(self, y_test, y_pred, p=1):
        super().__init__(y_test, y_pred)
        self.p = p

    def __call__(self):
        self.minkowski_distance()

    # 闵可夫斯基距离
    def minkowski_distance(self):
        i = 0
        sum = 0
        while i < len(self.y_true):
            sum += abs(self.y_true[i] - self.y_pred[i]) ** self.p
            i += 1
        if self.p == 1:
            print(sum)
        elif self.p == 2:
            print(sum ** 0.5)


# # 闵可夫斯基距离示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.cluster import KMeans
# 
#     # '''
#     # 对本例目前使用的鸢尾花数据集不适用，若要使用该数据集，需要先对target数据进行处理（str->int）
#     # '''
#     # # 鸢尾花数据集
#     # # 数据载入
#     # iris = pd.read_csv('../data/Iris.csv')
#     # # print(iris.head(10))
#     # # 数据分割
#     # x = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
#     # y = iris['Species'].values
#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # # print(x_train, x_test, y_train, y_test)
#     # # 模型训练、预测
#     # model = KMeans(n_clusters=3)
#     # model.fit(x_train, y_train)
#     # train_predict = model.predict(x_train)
#     # test_predict = model.predict(x_test)
#     # # # 模型评估
#     # distance_train = Distance(y_train, train_predict)
#     # distance_test = Distance(y_test, test_predict)
#     # distance_train()
#     # distance_test()
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
#     model = KMeans(n_clusters=3)
#     model.fit(x_train, y_train)
#     train_predict = model.predict(x_train)
#     test_predict = model.predict(x_test)
#     # # 模型评估
#     distance_train = Distance(y_train, train_predict)
#     distance_test = Distance(y_test, test_predict)
#     distance_train()
#     distance_test()
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
#     model = KMeans(n_clusters=3)
#     model.fit(x_train, y_train)
#     train_predict = model.predict(x_train)
#     test_predict = model.predict(x_test)
#     # # 模型评估
#     distance_train = Distance(y_train, train_predict)
#     distance_test = Distance(y_test, test_predict)
#     distance_train()
#     distance_test()
