import Splitter
import random as r
import pandas as pd

        
# 自助法
class BootStrapping(Splitter):
    def __init__(self, dataset):
        self.data: pd.DataFrame = dataset

    def split(self):
        data_len = len(self.data)
        index = list(range(0, data_len))
        train_index = []
        # 进行m次放回抽样，得到D'
        for i in range(data_len):
            train_index.append(r.randint(0, data_len-1))
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
# data = pd.read_csv('Iris.csv')
# BootStrapping(data).split()
