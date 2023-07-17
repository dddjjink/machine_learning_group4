from Evaluation import Evaluation


# 准确率
class Accuracy(Evaluation):
    def __init__(self, y_test, y_pred):
        super().__init__(y_test, y_pred)
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def __call__(self, *args, **kwargs):
        self.accuracy_cal()
        self.evaluate()

    def accuracy_cal(self):
        for true, pred in zip(self.y_true, self.y_pred):
            if true == pred:
                if true == 1:
                    self.true_positives += 1
                else:
                    self.true_negatives += 1
            else:
                if true == 1:
                    self.false_negatives += 1
                else:
                    self.false_positives += 1

    def evaluate(self):
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        accuracy = (self.true_positives + self.true_negatives) / total
        print(accuracy)


# # 准确率示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import LogisticRegression
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
#     clf = LogisticRegression(random_state=0, solver='lbfgs')
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     # 模型评估
#     acc_train = Accuracy(y_train, train_predict)
#     acc_test = Accuracy(y_test, test_predict)
#     acc_train()
#     acc_test()
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
#     clf = LogisticRegression(random_state=0, solver='lbfgs')
#     clf.fit(x_train, y_train)
#     train_predict = clf.predict(x_train)
#     test_predict = clf.predict(x_test)
#     # 模型评估
#     acc_train = Accuracy(y_train, train_predict)
#     acc_test = Accuracy(y_test, test_predict)
#     acc_train()
#     acc_test()


