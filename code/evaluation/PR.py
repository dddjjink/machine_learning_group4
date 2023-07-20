import numpy as np
from matplotlib import pyplot as plt
from .Evaluation import Evaluation


# P-R曲线，可视化
class PR(Evaluation):
    def __init__(self, y_true, y_score):
        self.y_true = y_true
        self.y_score = y_score

    def __call__(self, *args, **kwargs):
        self.plot()

    def precision_recall_curve(self):
        # 计算不同阈值下的精确率和召回率
        thresholds = np.unique(self.y_score)
        precisions = []
        recalls = []
        for threshold in thresholds:
            y_pred = np.where(self.y_score >= threshold, 1, 0)
            tp = np.sum(np.logical_and(self.y_true == 1, y_pred == 1))
            fp = np.sum(np.logical_and(self.y_true == 0, y_pred == 1))
            fn = np.sum(np.logical_and(self.y_true == 1, y_pred == 0))
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls

    def plot(self):
        # 绘制PR曲线
        precisions, recalls = self.precision_recall_curve()
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()


# # P-R曲线示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import LogisticRegression
# 
#     # '''
#     # 对本例的鸢尾花数据集不适用，P-R曲线适用二分类问题
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
#     # y_score_train = clf.decision_function(x_train)
#     # y_score_test = clf.decision_function(x_test)
#     # # 模型评估
#     # PR_train = PR(y_train, y_score_train)
#     # PR_test = PR(y_test, y_score_test)
#     # PR_train()
#     # PR_test()
#
#     # '''
#     # 对本例的红酒数据集不适用，P-R曲线适用二分类问题
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
#     # y_score_train = clf.decision_function(x_train)
#     # y_score_test = clf.decision_function(x_test)
#     # # 模型评估
#     # PR_train = PR(y_train, y_score_train)
#     # PR_test = PR(y_test, y_score_test)
#     # PR_train()
#     # PR_test()
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
#     y_score_train = clf.decision_function(x_train)
#     y_score_test = clf.decision_function(x_test)
#     # 模型评估
#     PR_train = PR(y_train, y_score_train)
#     PR_test = PR(y_test, y_score_test)
#     PR_train()
#     PR_test()
