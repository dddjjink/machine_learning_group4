from Evaluation import Evaluation


# F1度量
class F1(Evaluation):
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def __call__(self, *args, **kwargs):
        self.f1_score()
        self.evaluate()

    def f1_score(self):
        for true, pred in zip(self.y_true, self.y_pred):
            if true == pred:
                if true == 1:
                    self.true_positives += 1
            else:
                if true == 1:
                    self.false_negatives += 1
                else:
                    self.false_positives += 1

    def evaluate(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f1_score)


# # F1度量示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import LogisticRegression
# 
#     # '''
#     # 对本例的鸢尾花数据集不适用，F1度量适用二分类问题
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
#     # f1_train = F1(y_train, train_predict)
#     # f1_test = F1(y_test, test_predict)
#     # f1_train()
#     # f1_test()
# 
#     # '''
#     # 对本例的红酒数据集不适用，F1度量适用二分类问题
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
#     # f1_train = F1(y_train, train_predict)
#     # f1_test = F1(y_test, test_predict)
#     # f1_train()
#     # f1_test()
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
#     f1_train = F1(y_train, train_predict)
#     f1_test = F1(y_test, test_predict)
#     f1_train()
#     f1_test()
