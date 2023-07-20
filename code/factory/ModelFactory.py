from Factory.Factory import Factory
from model.DecisionTree import DecisionTree
from model.GradientBoosting import GBDT
from model.KMeans import KMeans
from model.KNN import KNN
from model.LR import LinearRegression
from model.LogisticRegression import LogisticRegression
from model.NB import NB
from model.RandomForest import RandomForest
from model.SVM import SVM


class ModelFactory(Factory):
    @staticmethod
    def create_model(model):
        if model =='KNN':
            return KNN
        elif model =='K_means':
            return KMeans
        elif model=='Decision Tree':
            return DecisionTree
        elif model=='GradientBoosting':
            return GBDT
        elif model=='LR':
            return LinearRegression
        elif model=='SVM':
            return SVM
        elif model=='LogisticRegression':
            return LogisticRegression
        elif model=='NB':
            return NB
        elif model=='Random Forest':
            return RandomForest
