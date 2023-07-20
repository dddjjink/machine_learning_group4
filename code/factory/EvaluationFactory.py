from Factory.Factory import Factory
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


class EvaluationFactory(Factory):
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

