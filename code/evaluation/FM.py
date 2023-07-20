import numpy as np
from .Evaluation import Evaluation


class FM(Evaluation):
    def __init__(self, labels_true, labels_pred):
        super().__init__(labels_true, labels_pred)

    def __call__(self, *args, **kwargs):
        return self.compute_fm_index()

    def compute_confusion_matrix(self):
        n_samples = len(self.y_true)
        confusion_matrix = np.zeros((2, 2))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if self.y_true[i] == self.y_true[j] and self.y_pred[i] == self.y_pred[j]:
                    confusion_matrix[0, 0] += 1  # true positive
                elif self.y_true[i] != self.y_true[j] and self.y_pred[i] != self.y_pred[j]:
                    confusion_matrix[1, 1] += 1  # true negative
                elif self.y_true[i] == self.y_true[j] and self.y_pred[i] != self.y_pred[j]:
                    confusion_matrix[0, 1] += 1  # false negative
                else:
                    confusion_matrix[1, 0] += 1  # false positive
        return confusion_matrix

    def compute_fm_index(self):
        cm = self.compute_confusion_matrix()
        tp = cm[0, 0]
        fp = cm[1, 0]
        fn = cm[0, 1]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        fm_index = np.sqrt(precision * recall)

        print(round(fm_index, 2))
        return round(fm_index, 2)


# # FM示例用法
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
#     model = KMeans(n_clusters=3)
#     model.fit(x_train, y_train)
#     train_predict = model.predict(x_train)
#     test_predict = model.predict(x_test)
#     # # 模型评估
#     fm_train = FM(y_train, train_predict)
#     fm_test = FM(y_test, test_predict)
#     fm_train()
#     fm_test()
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
#     fm_train = FM(y_train, train_predict)
#     fm_test = FM(y_test, test_predict)
#     fm_train()
#     fm_test()
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
#     fm_train = FM(y_train, train_predict)
#     fm_test = FM(y_test, test_predict)
#     fm_train()
#     fm_test()
