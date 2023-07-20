from data.Dataset import IrisDataset, WineQualityDataset, HeartDiseaseDataSet
from splitter.BootStrapping import BootStrapping
from splitter.HoldOut import HoldOut
from model.DecisionTree import DecisionTree
from model.GradientBoosting import GBDT
from model.KMeans import KMeans
from model.KNN import KNN
from model.LR import LinearRegression
from model.LogisticRegression import LogisticRegression
from model.NB import NB
from model.RandomForest import RandomForest
from model.SVM import SVM
from evaluation.AUC import AUC
from evaluation.Accuracy import Accuracy
from evaluation.Distance import Distance
from evaluation.F1 import F1
from evaluation.FM import FM
from evaluation.MSE import MSE
from evaluation.PR import PR
from evaluation.RMSE import RMSE
from evaluation.ROC import ROC
from evaluation.Rand import Rand


class Factory:
    pass


class DataFactory(Factory):
    @staticmethod
    def create_dataset(dataset):
        if dataset == 'iris':
            return IrisDataset()
        elif dataset == 'wine':
            return WineQualityDataset()
        elif dataset == 'heart':
            return HeartDiseaseDataSet()


class SplitterFactory(Factory):
    @staticmethod
    def create_splitter(splitter, X,y,percent):
        if splitter == 'bootstraping':
            return BootStrapping(X,y)
        elif splitter == 'holdout':
            return HoldOut(X,y,percent)
            
class ModelFactory(Factory):
    @staticmethod
    def create_model(model):
        if model == 'KNN':
            return KNN()
        elif model == 'K_means':
            return KMeans()
        elif model == 'Decision Tree':
            return DecisionTree()
        elif model == 'GradientBoosting':
            return GBDT()
        elif model == 'LR':
            return LinearRegression()
        elif model == 'SVM':
            return SVM()
        elif model == 'LogisticRegression':
            return LogisticRegression()
        elif model == 'NB':
            return NB()
        elif model == 'Random Forest':
            return RandomForest()


class EvaluationFactory:
    @staticmethod
    def create_evaluation(evaluation, y_true, y_pred):
        if evaluation == 'auc':
            return AUC(y_true, y_pred)
        elif evaluation == 'accuracy':
            return Accuracy(y_true, y_pred)
        elif evaluation == 'distance':
            return Distance(y_true, y_pred)
        elif evaluation == 'f1':
            return F1(y_true, y_pred)
        elif evaluation == 'fm':
            return FM(y_true, y_pred)
        elif evaluation == 'mse':
            return MSE(y_true, y_pred)
        elif evaluation == 'pr':
            return PR(y_true, y_pred)
        elif evaluation == 'rmse':
            return RMSE(y_true, y_pred)
        elif evaluation == 'roc':
            return ROC(y_true, y_pred)
        elif evaluation == 'rand':
            return Rand(y_true, y_pred)
