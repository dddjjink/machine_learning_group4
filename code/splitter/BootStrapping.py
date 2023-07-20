from .Splitter import Splitter
import random as r


# 自助法
class BootStrapping(Splitter):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __call__(self, *args, **kwargs):
        self.split()

    def split(self):
        data_len = len(self.data)
        index = list(range(0, data_len))
        train_index = []
        # 进行m次放回抽样，得到D'
        for i in range(data_len):
            train_index.append(r.randint(0, data_len - 1))
        # 得到D\D'
        test_index = list(set(index).difference(set(train_index)))
        # 产生训练/测试集
        train_data_df = self.data[train_index]
        test_data_df = self.data[test_index]
        train_target_df = self.target[train_index]
        test_target_df = self.target[test_index]
        # # 打印结果进行验证
        # print("\n\ntrain data set: \n", len(train_data_df), "\n", train_data_df)
        # print("\n\ntest data set: \n", len(test_data_df), "\n", test_data_df)
        # print("\n\ntrain target set: \n", len(train_target_df), "\n", train_target_df)
        # print("\n\ntest target set: \n", len(test_target_df), "\n", test_target_df)
        return train_data_df, test_data_df, train_target_df, test_target_df
        # x_train, x_test, y_train, y_test


# # 自助法示例用法
# if __name__ == '__main__':
#     import pandas as pd
#
#     # 鸢尾花数据集
#     iris = pd.read_csv('../data/Iris.csv')
#     # print(iris.head(10))
#     # 数据分割
#     x = iris.drop(['Species', 'Id'], axis=1).values
#     y = iris['Species'].values
#     data_split = BootStrapping(x, y)
#     x_train, x_test, y_train, y_test = data_split.split()
#     print(x_train, x_test, y_train, y_test)
#
#     # 红酒数据集
#     wine = pd.read_csv('../data/WineQT.csv')
#     # print(wine.head(10))
#     # 数据分割
#     x = wine.drop(['quality', 'Id'], axis=1).values
#     y = wine['quality'].values
#     data_split = BootStrapping(x, y)
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
#     data_split = BootStrapping(x, y)
#     x_train, x_test, y_train, y_test = data_split.split()
#     # print(x_train, x_test, y_train, y_test)

