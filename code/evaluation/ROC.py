import numpy as np
import matplotlib.pyplot as plt
from Evaluation import Evaluation


# ROC曲线，可视化
class ROC(Evaluation):
    def __init__(self, y_test, y_pred):
        super().__init__(y_test, y_pred)
        self.tpr = []
        self.fpr = []

    def __call__(self, *args, **kwargs):
        self.plot()

    def roc_curve(self):
        true_positives = 0
        false_positives = 0
        total_positives = np.sum(self.y_true)
        total_negatives = len(self.y_true) - total_positives

        for true, pred in zip(self.y_true, self.y_pred):
            if true == pred and true == 1:
                true_positives += 1
            elif true != pred and true == 0:
                false_positives += 1

            tpr = true_positives / total_positives
            fpr = false_positives / total_negatives

            self.tpr.append(tpr)
            self.fpr.append(fpr)
        return self.tpr, self.fpr

    def plot(self):
        tpr, fpr = self.roc_curve()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()


# # ROC曲线示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import LogisticRegression
# 
#     # '''
#     # 对本例的鸢尾花数据集不适用，ROC曲线适用二分类问题
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
#     # clf = LogisticRegression(random_state=0, solver='lbfgs')
#     # clf.fit(x_train, y_train)
#     # train_predict = clf.predict(x_train)
#     # test_predict = clf.predict(x_test)
#     # # 模型评估
#     # ROC_train = ROC(y_train, train_predict)
#     # ROC_test = ROC(y_test, test_predict)
#     # ROC_train()
#     # ROC_test()
# 
#     # '''
#     # 对本例的红酒数据集不适用，ROC曲线适用二分类问题
#     # '''
#     # # 红酒数据集
#     # # 数据载入
#     # wine = pd.read_csv('../data/WineQT.csv')
#     # # print(wine.head(10))
#     # # 数据分割
#     # x = wine.drop(['quality', 'Id'], axis=1).values
#     # y = wine['quality'].values
#     # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)
#     # # print(x_train, x_test, y_train, y_test)
#     # # 模型训练、预测
#     # clf = LogisticRegression(random_state=0, solver='lbfgs')
#     # clf.fit(x_train, y_train)
#     # train_predict = clf.predict(x_train)
#     # test_predict = clf.predict(x_test)
#     # # 模型评估
#     # ROC_train = ROC(y_train, train_predict)
#     # ROC_test = ROC(y_test, test_predict)
#     # ROC_train()
#     # ROC_test()
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
#     clf = LogisticRegression(random_state=0, solver='lbfgs')
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     # 模型评估
#     ROC_train = ROC(y_train, train_predict)
#     ROC_test = ROC(y_test, test_predict)
#     ROC_train()
#     ROC_test()
