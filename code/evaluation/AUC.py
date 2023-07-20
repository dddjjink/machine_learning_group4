from .Evaluation import Evaluation
from .ROC import ROC
import numpy as np


# AUC
class AUC(Evaluation):
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)
        self.tpr = []
        self.fpr = []

    def __call__(self, *args, **kwargs):
        self.auc()
        return self.evaluate()

    def auc(self):
        roc_curve = ROC(self.y_true, self.y_pred)
        roc_curve.roc_curve()
        tpr, fpr = roc_curve.tpr, roc_curve.fpr

        self.tpr.extend(tpr)
        self.fpr.extend(fpr)
        return self.tpr, self.fpr

    def evaluate(self):
        sorted_indices = np.argsort(self.fpr)
        sorted_fpr = np.array(self.fpr)[sorted_indices]
        sorted_tpr = np.array(self.tpr)[sorted_indices]
        auc = np.trapz(sorted_tpr, sorted_fpr)
        print(round(auc, 2))
        return round(auc, 2)


# # AUC示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import LogisticRegression
# 
#     # '''
#     # 对本例的鸢尾花数据集不适用，AUC适用二分类问题
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
#     # auc_train = AUC(y_train, train_predict)
#     # auc_test = AUC(y_test, test_predict)
#     # auc_train()
#     # auc_test()
# 
#     # '''
#     # 对本例的红酒数据集不适用，AUC适用二分类问题
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
#     # auc_train = AUC(y_train, train_predict)
#     # auc_test = AUC(y_test, test_predict)
#     # auc_train()
#     # auc_test()
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
#     auc_train = AUC(y_train, train_predict)
#     auc_test = AUC(y_test, test_predict)
#     auc_train()
#     auc_test()
