import os
import pandas as pd


class Dataset:
    def __init__(self, path) -> None:
        self.path = path
        self.dataset: pd.DataFrame = None
        self.data = None
        self.target = None

    # 使用某个数据集时再加载
    def load(self):
        self.dataset = pd.read_csv(self.path)

    # 对数据集分x和y
    def data_target_split(self, x, y):
        self.data = x
        self.target = y


# 鸢尾花数据集
class IrisDataset(Dataset):
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), 'Iris.csv')
        super().__init__(path)

    def data_target(self):
        super().load()
        x = self.dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = self.dataset['Species'].values
        super().data_target_split(x, y)


# 红酒数据集
class WineQualityDataset(Dataset):
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), 'WineQT.csv')
        super().__init__(path)

    def data_target(self):
        super().load()
        x = self.dataset.drop(['quality', 'Id'], axis=1).values
        y = self.dataset['quality'].values
        super().data_target_split(x, y)


# 心脏病数据集，二分类数据集，适用逻辑回归、SVM、决策树
class HeartDiseaseDataSet(Dataset):
    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), 'heart.csv')
        super().__init__(path)

    def data_target(self):
        super().load()
        x = self.dataset.drop(['target'], axis=1).values
        y = self.dataset['target'].values
        super().data_target_split(x, y)


# # 设定数据集对象
# if __name__ == '__main__':
#     iris = IrisDataset()
#     iris.data_target()
#     print(iris.data)
#     wine = WineQualityDataset()
#     wine.data_target()
#     print(wine.target)
#     heart = HeartDiseaseDataSet()
#     heart.data_target()
#     print(heart.data)
