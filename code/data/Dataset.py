import pandas as pd


class Dataset:
    def __init__(self, path) -> None:
        self.path = path
        self.data: pd.DataFrame = None

    # 使用某个数据集时再加载
    def load(self):
        self.data = pd.read_csv(self.path)


# 鸢尾花数据集
class IrisDataset(Dataset):
    def __init__(self):
        super().__init__('Iris.csv')


# 红酒数据集
class WineQualityDataset(Dataset):
    def __init__(self):
        super().__init__('WineQT.csv')


# 心脏病数据集，二分类数据集，适用逻辑回归、SVM、决策树
class HeartDiseaseDataSet(Dataset):
    def __init__(self):
        super().__init__('heart.csv')


# # 设定数据集对象
# if __name__ == '__main__':
#     iris = IrisDataset()
#     iris.load()
#     print(iris.data.iloc[149])
#     wine = WineQualityDataset()
#     wine.load()
#     print(wine.data)
#     heart = HeartDiseaseDataSet()
#     heart.load()
#     print(heart.data)
