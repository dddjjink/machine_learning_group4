import Evaluation


# AUC
class AUC(Evaluation):
    def __init__(self, y_true, y_pred):
        self.tpr = []
        self.fpr = []
        self.y_true = y_true
        self.y_pred = y_pred
    def AUC(self):
        roc_curve = ROC(self.y_true, self.y_pred)
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
