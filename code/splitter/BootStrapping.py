from Splitter import Splitter
import random as r


# 自助法
class BootStrapping(Splitter):
    def __init__(self, dataset):
        super().__init__(dataset)

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
        train_df = self.data.iloc[train_index]
        test_df = self.data.iloc[test_index]
        # # 打印结果进行验证
        # print("data set: \n", self.data)
        # print("\n\ntrain set: \n", train_df)
        # print("\n\ntest set: \n", test_df)
        return train_df, test_df


# # 自助法示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     # 鸢尾花数据集
#     iris = pd.read_csv('../data/Iris.csv')
#     print(iris.head(10))
#     data_split = BootStrapping(iris)
#     train_data, test_data = data_split.split()
#     print(train_data, test_data)
#     # 红酒数据集
#     wine = pd.read_csv('../data/WineQT.csv')
#     print(wine.head(10))
#     data_split = BootStrapping(wine)
#     train_data, test_data = data_split.split()
#     print(train_data, test_data)

