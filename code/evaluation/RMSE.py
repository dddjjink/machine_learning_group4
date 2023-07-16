import Evaluation


# 均方根误差
class RMSE(Evaluation):
    def __str__(self):
        return 'MSE'

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def loss(self, y, y_pred):
        return math.sqrt(0.5 * np.sum((y_pred - y) ** 2, axis=-1))
