from Splitter import Splitter


# 交叉验证法
class CV(Splitter):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __call__(self, *args, **kwargs):
        self.split()

    def split(self):
        data_set = []
        data_len = len(self.data)
        # k值
        k = 10
        for i in range(k):
            tmp = []
            j = i
            while j < data_len:
                tmp.append(self.data.iloc[j])
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

        return train_set, test_set


# # 交叉验证法示例用法
# if __name__ == '__main__':
#     import pandas as pd
#     data = pd.read_csv('../data/Iris.csv')
#     print(data.head(10))
#     data_split = CV(data)
#     train_data, test_data = data_split.split()
#     print(train_data, test_data)
