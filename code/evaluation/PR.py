import Evaluation


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
