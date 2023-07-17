from Splitter import Splitter
import random as r


# 留出法
class HoldOut(Splitter):
    def __init__(self, dataset, test_size=0.2):
        super().__init__(dataset)
        self.split_size = test_size

    def __call__(self, *args, **kwargs):
        self.split()

    def split(self):
        # 计算测试集的数量
        test_num = int(len(self.data) * self.split_size)

        # 打乱数据集
        '''
        !!!!!!!!!!此处有误!!!!!!!!!!
        '''
        r.shuffle(self.data)

        # 划分数据集
        train_data = self.data[:-test_num]
        test_data = self.data[-test_num:]

        return train_data, test_data


# # 留出法示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     data = pd.read_csv('../data/Iris.csv')
#     print(data.head(10))
#     data_split = HoldOut(data)
#     train_data, test_data = data_split.split()
#     print(train_data, test_data)
