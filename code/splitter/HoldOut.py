import random as r
import Splitter


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


# # 留出法示例用法
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# test_size = 0.2
# train_data, test_data = train_test_split(data, test_size)
# print("训练集:", train_data)
# print("测试集:", test_data)
