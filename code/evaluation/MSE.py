import Evaluation


# 均方误差
class MSE(Evaluation):
    def __str__(self):
        return 'MSE'

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def loss(self, y, y_pred):
        return 0.5 * np.sum((y_pred - y) ** 2, axis=-1)
