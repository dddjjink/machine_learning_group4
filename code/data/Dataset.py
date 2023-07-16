import pandas as pd


class Dataset:
    def __init__(self, path) -> None:
        self.path = path
        self.data: pd.DataFrame = None

    def load(self):
        self.data = pd.read_csv(self.path)


# 鸢尾花数据集
class IrisDataset(Dataset):
    def __init__(self):
        super().__init__('Iris.csv')
        super().load()


# 红酒数据集
class WineQualityDataset(Dataset):
    def __init__(self):
        super().__init__('WineQT.csv')
        super().load()


# # 设定数据集对象
# IrisDataset()
# print(IrisDataset().data.iloc[149])
# WineQualityDataset()
# print(WineQualityDataset().data)
