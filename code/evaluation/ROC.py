import numpy as np
import matplotlib.pyplot as plt
import Evaluation


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
