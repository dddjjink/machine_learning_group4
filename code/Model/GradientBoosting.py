import CART_regression_tree
import numpy as np


def get_init_val(y):
    return sum(y) / len(y)


def get_residuals(y, y_hat):
    y_residuals = []
    for i in range(len(y)):
        y_residuals.append(y[i] - y_hat[i])
    return y_residuals


def GBDT_cal_err(Y_test, predicts):
    y_test = np.array(Y_test)
    pre_y = np.array(predicts)
    error = np.square(y_test - pre_y).sum() / len(Y_test)
    return error


class GBDT(object):
    def __init__(self):
        self.trees = None  # 存储生成的多棵回归树
        self.learn_rate = None  # 学习率
        self.init_val = None  # 初始值

    def fit(self, X, Y, n_estimates, learn_rate, min_sample, min_err):
        self.trees = []
        # 获取初始值
        self.init_val = get_init_val(Y)
        n = len(Y)
        # 以样本标签的均值作为初始预测值
        y_hat = [self.init_val] * n
        # 生成初始残差
        y_residuals = get_residuals(Y, y_hat)

        self.learn_rate = learn_rate
        for k in range(n_estimates):
            # 拟合残差生成CART树
            tree = CART_regression_tree.CART_regression(X, y_residuals, min_sample, min_err)
            for i in range(len(X)):
                res_hat = CART_regression_tree.predict(X[i], tree)
                y_hat[i] += self.learn_rate * res_hat
            y_residuals = get_residuals(Y, y_hat)
            self.trees.append(tree)

    def GBDT_predict(self, X_test):
        predicts = []
        for i in range(len(X_test)):
            pre_y = self.init_val
            for tree in self.trees:
                pre_y += CART_regression_tree.predict(X_test[i], tree)
            predicts.append(pre_y)
        return predicts
