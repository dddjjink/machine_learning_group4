class Evaluation:
    pass


# 分类评价
# 准确率
class Accuracy(Evaluation):
    def __init__(self, y_true, y_pred):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.y_true = y_true
        self.y_pred = y_pred
    def Accuracy(self):
        for true, pred in zip(self.y_true, self.y_pred):
            if true == pred:
                if true == 1:
                    self.true_positives += 1
                else:
                    self.true_negatives += 1
            else:
                if true == 1:
                    self.false_negatives += 1
                else:
                    self.false_positives += 1

    def evaluate(self):
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        accuracy = (self.true_positives + self.true_negatives) / total
        return accuracy



# F1度量
class F1(Evaluation):
    def __init__(self, y_true, y_pred):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.y_true = y_true
        self.y_pred = y_pred
    def F1Score(self):
        for true, pred in zip(self.y_true, self.y_pred):
            if true == pred:
                if true == 1:
                    self.true_positives += 1
            else:
                if true == 1:
                    self.false_negatives += 1
                else:
                    self.false_positives += 1

    def evaluate(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


# P-R曲线，可视化
class PR(Evaluation):
    def __init__(self, y_true, y_pred):
        self.precision = []
        self.recall = []
        self.y_true = y_true
        self.y_pred = y_pred
    def PRCurve(self):
        true_positives = 0
        false_positives = 0
        total_positives = np.sum(self.y_true)

        for true, pred in zip(self.y_true, self.y_pred):
            if true == pred and true == 1:
                true_positives += 1
            elif true != pred and true == 1:
                false_positives += 1

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / total_positives

            self.precision.append(precision)
            self.recall.append(recall)
            return self.precision, self.recall
    def plot(self):
        precision, recall = self.PRCurve()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('P-R Curve')
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
# ROC曲线，可视化
class ROC(Evaluation):
    def __init__(self, y_true, y_pred):
        self.tpr = []
        self.fpr = []
        self.y_true = y_true
        self.y_pred = y_pred
    def ROCCurve(self):
        true_positives = 0
        false_positives = 0
        total_positives = np.sum(self.y_true)
        total_negatives = len(self.y_true) - total_positives

        for true, pred in zip(self.y_true, self.y_pred):
            if true == pred and true == 1:
                true_positives += 1
            elif true != pred and true == 0:
                false_positives += 1

            tpr = true_positives / total_positives
            fpr = false_positives / total_negatives

            self.tpr.append(tpr)
            self.fpr.append(fpr)
        return self.tpr, self.fpr
    def plot(self):
        tpr, fpr = self.ROCCurve()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()


# AUC
class AUC(Evaluation):
    def __init__(self, y_true, y_pred):
        self.tpr = []
        self.fpr = []
        self.y_true = y_true
        self.y_pred = y_pred
    def AUC(self):
        roc_curve = ROCCurve(self.y_true, self.y_pred)
        roc_curve.ROCCurve()
        tpr, fpr = roc_curve.tpr,roc_curve.fpr

        self.tpr.extend(tpr)
        self.fpr.extend(fpr)
        return self.tpr, self.fpr
    def evaluate(self):
        sorted_indices = np.argsort(self.fpr)
        sorted_fpr = np.array(self.fpr)[sorted_indices]
        sorted_tpr = np.array(self.tpr)[sorted_indices]
        auc = np.trapz(sorted_tpr, sorted_fpr)
        return auc


# 回归评价
# 均方误差
class MSE(Evaluation):
    pass


# 均方根误差
class RMSE(Evaluation):
    pass


# 聚类指数，外部指标
class FM(Evaluation):
    pass


class Rand(Evaluation):
    pass


# 内部指标
class DB(Evaluation):
    pass


# 距离公式
class Distance(Evaluation):
    pass


# 惯性
class Inertia(Evaluation):
    pass


# K均值聚类算法
class KMeans(Evaluation):
    pass
