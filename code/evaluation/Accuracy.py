import Evaluation


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
