from Factory.Factory import Factory
from data.Dataset import IrisDataset, WineQualityDataset, HeartDiseaseDataSet


class DataFactory(Factory):
    @staticmethod
    def create_dataset(dataset):
        if dataset == 'iris':
            return IrisDataset()
        elif dataset == 'wine':
            return WineQualityDataset()
        elif dataset == 'heart':
            return HeartDiseaseDataSet()
