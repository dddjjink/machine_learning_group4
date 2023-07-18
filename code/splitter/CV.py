from Splitter import Splitter


# 交叉验证法
class CV(Splitter):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.data_set = []
        self.target_set = []
        self.train_data_set = []
        self.test_data_set = []
        self.train_target_set = []
        self.test_target_set = []

    def __call__(self, *args, **kwargs):
        self.split()

    def split(self):
        data_len = len(self.data)
        # k值
        k = 10
        for i in range(k):
            tmp_data = []
            tmp_target = []
            j = i
            while j < data_len:
                tmp_data.append(self.data[j])
                tmp_target.append(self.target[j])
                j = j + k
            self.data_set.append(tmp_data)
            self.target_set.append(tmp_target)

        for i in range(k):
            self.test_data_set = self.data_set[i]
            self.test_target_set = self.target_set[i]
            for j in range(k):
                if i != j:
                    self.train_data_set.append(self.data_set[j])
                    self.train_target_set.append(self.target_set[j])
            # for j in range(k-1):
            #     print(train_set[j])
            # print(test_set)

        # # 打印结果进行验证
        # print("\n\ntrain data set: \n", len(self.train_data_set), "\n", self.train_data_set)
        # print("\n\ntest data set: \n", len(self.test_data_set), "\n", self.test_data_set)
        # print("\n\ntrain target set: \n", len(self.train_target_set), "\n", self.train_target_set)
        # print("\n\ntest target set: \n", len(self.test_target_set), "\n", self.test_target_set)
        return self.train_data_set, self.test_data_set, self.train_target_set, self.test_target_set
        # x_train, x_test, y_train, y_test


# # 交叉验证法示例用法
# if __name__ == '__main__':
#     import pandas as pd
# 
#     # 鸢尾花数据集
#     iris = pd.read_csv('../data/Iris.csv')
#     # print(iris.head(10))
#     # 数据分割
#     x = iris.drop(['Species', 'Id'], axis=1).values
#     y = iris['Species'].values
#     data_split = CV(x, y)
#     x_train, x_test, y_train, y_test = data_split.split()
#     # print(x_train, x_test, y_train, y_test)
# 
#     # 红酒数据集
#     wine = pd.read_csv('../data/WineQT.csv')
#     # print(wine.head(10))
#     # 数据分割
#     x = wine.drop(['quality', 'Id'], axis=1).values
#     y = wine['quality'].values
#     data_split = CV(x, y)
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
#     data_split = CV(x, y)
#     x_train, x_test, y_train, y_test = data_split.split()
#     # print(x_train, x_test, y_train, y_test)
