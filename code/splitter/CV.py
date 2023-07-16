import splitter
import pandas as pd


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


# data = pd.read_csv('Iris.csv')
# CV(data).k_fold()
