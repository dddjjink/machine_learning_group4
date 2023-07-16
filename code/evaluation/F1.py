import Evaluation


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
