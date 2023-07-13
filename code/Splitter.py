import random as r
import pandas as pd


class Splitter:
    # def __init__(self, rate):
    pass


# 留出法
class HoldOut(Splitter):
    def train_test_split(data, test_size):
        """
        将数据集划分为训练集和测试集
        :param data: 原始数据集
        :param test_size: 测试集所占比例
        :return: 训练集和测试集
        """

        # 计算测试集的数量
        test_num = int(len(data) * test_size)

        # 打乱数据集
        r.shuffle(data)
        # 划分数据集
        train_data = data[:-test_num]
        test_data = data[-test_num:]

        return train_data, test_data


# 交叉验证法
class CV(Splitter):
    def __init__(self, dataset):
        self.data: pd.DataFrame = dataset

    def k_fold(self):
        data=pd.DataFrame(self.data)
        data_set = []
        data_len = len(self.data)
        # k值
        k = 10  # per想代表训练集的比例
        for i in range(k):
            tmp = []
            j = i
            while j < data_len:
                tmp.append(data.iloc[j])
                j = j + k
            data_set.append(tmp)

        for i in range(k):
            test_set = data_set[i]
            train_set = []
            for j in range(k):
                if i != j:
                    train_set.append(data_set[j])
            # for j in range(k-1):
            #     print(train_set[j])
            # print(test_set)

        return train_set,test_set
        
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


# # 留出法示例用法
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# test_size = 0.2
# train_data, test_data = train_test_split(data, test_size)
# print("训练集:", train_data)
# print("测试集:", test_data)

# # 自助法示例用法
# data = pd.read_csv('Iris.csv')
# BootStrapping(data).split()

# data = pd.read_csv('Iris.csv')
# CV(data).k_fold()
