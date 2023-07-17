import numpy as np
from matplotlib import pyplot as plt
from Evaluation import Evaluation


# P-R曲线，可视化
class PR(Evaluation):
    def __init__(self, y_test, y_pred):
        super().__init__(y_test, y_pred)
        self.precision = []
        self.recall = []

    def __call__(self, *args, **kwargs):
        self.plot()

    def pr_curve(self):
        true_positives = 0
        false_positives = 0
        total_positives = np.sum(self.y_true)

        for true, pred in zip(self.y_true, self.y_pred):
            if true == pred and true == 1:
                true_positives += 1
            elif true != pred and true == 1:
                false_positives += 1

            '''
            !!!!!!!!!!此处有误!!!!!!!!!!
            错误原因：ZeroDivisionError: division by zero
            '''
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / total_positives

            self.precision.append(precision)
            self.recall.append(recall)
            return self.precision, self.recall

    def plot(self):
        precision, recall = self.pr_curve()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('P-R Curve')
        plt.show()


# # P-R曲线示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import LogisticRegression
# 
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
#     # PR_train = PR(y_train, train_predict)
#     # PR_test = PR(y_test, test_predict)
#     # PR_train()
#     # PR_test()
# 
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
#     # PR_train = PR(y_train, train_predict)
#     # PR_test = PR(y_test, test_predict)
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
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     # 模型评估
#     PR_train = PR(y_train, train_predict)
#     PR_test = PR(y_test, test_predict)
#     PR_train()
#     PR_test()
