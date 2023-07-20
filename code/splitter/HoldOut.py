from .Splitter import Splitter
import random as r


# 留出法
class HoldOut(Splitter):
    def __init__(self, x, y, test_size="0.2"):
        super().__init__(x, y)
        self.split_size = eval(test_size)
        self.random_state = None

    def __call__(self, *args, **kwargs):
        self.split()

    def split(self):
        # 设置随机种子
        r.seed(self.random_state)
        # 获取样本数量和测试集大小
        n_samples = len(self.data)
        n_test = int(n_samples * self.split_size)
        # 随机选择测试集的索引
        test_indices = r.sample(range(n_samples), n_test)
        # 构建训练集和测试集
        train_index=[]
        test_index=[]
        for i in range(n_samples):
            if i not in test_indices:
                train_index.append(i)
        for i in test_indices:
            test_index.append(i)
        x_train = self.data[train_index]
        x_test = self.data[test_index]
        y_train = self.target[train_index]
        y_test = self.target[test_index]
        # # 打印结果进行验证
        # print("\n\ntrain data set: \n", len(x_train), "\n", x_train)
        # print("\n\ntest data set: \n", len(x_test), "\n", x_test)
        # print("\n\ntrain target set: \n", len(y_train), "\n", y_train)
        # print("\n\ntest target set: \n", len(y_test), "\n", y_test)
        return x_train, x_test, y_train, y_test
        # x_train, x_test, y_train, y_test


# # 留出法示例用法
# if __name__ == '__main__':
#     import pandas as pd
#
#     # 鸢尾花数据集
#     iris = pd.read_csv('../data/Iris.csv')
#     # print(iris.head(10))
#     # 数据分割
#     x = iris.drop(['Species', 'Id'], axis=1).values
#     y = iris['Species'].values
#     data_split = HoldOut(x, y)
#     x_train, x_test, y_train, y_test = data_split.split()
#     print(x_train)
# 
#     # 红酒数据集
#     wine = pd.read_csv('../data/WineQT.csv')
#     # print(wine.head(10))
#     # 数据分割
#     x = wine.drop(['quality', 'Id'], axis=1).values
#     y = wine['quality'].values
#     data_split = HoldOut(x, y)
#     x_train, x_test, y_train, y_test = data_split.split()
#     # print(x_train, x_test, y_train, y_test)
# 
#     # 心脏病数据集
#     # 数据载入
#     heart = pd.read_csv('../data/heart.csv')
#     # print(heart.head(10))
#     # 数据分割
#     x = heart.drop(['target'], axis=1).values
#     y = heart['target'].values
#     data_split = HoldOut(x, y)
#     x_train, x_test, y_train, y_test = data_split.split()
#     # print(x_train, x_test, y_train, y_test)
