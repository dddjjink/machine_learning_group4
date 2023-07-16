import copy
import numpy as np


class Node:
    def __init__(self, feature=None, split_val=None, results=None, left=None, right=None):
        self.feature = feature
        self.split_val = split_val
        self.results = results
        self.left = left
        self.right = right


def combine(X, Y):
    data = copy.deepcopy(X)
    for i in range(len(X)):
        data[i].append(Y[i])
    return data


def leaf(dataSet):
    data = np.array(dataSet)
    return np.mean(data[:, -1])


def err_cnt(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def split_tree(data, feature, split_val):
    set_L, set_R = [], []
    tmp_LX, tmp_LY, tmp_RX, tmp_RY = [], [], [], []
    for i in data:
        if data[feature] < split_val:
            tmp_LX.append(list(data[0:-1]))
            tmp_LY.append(list(data[-1]))
        else:
            tmp_RX.append(list(data[0:-1]))
            tmp_RY.append(list(data[-1]))
    set_L.append(tmp_LX)
    set_L.append(tmp_LY)
    set_R.append(tmp_RX)
    set_R.append(tmp_RY)
    return set_L, set_R


class CART_regression(object):
    def __init__(self, X, Y, min_sample, min_err):
        self.X = X
        self.Y = Y
        self.min_sample = min_sample
        self.min_err = min_err

    def fit(self):
        data = combine(self.X, self.Y)
        data = np.array(data)
        # 初始化
        bestErr = err_cnt(data)
        # 最佳切分值
        bestCreteria = None
        # 切分集合
        bestSets = None
        # 如果data样本数少于最小样本数或者样本误差小于最小误差，生成叶子节点
        if len(data) <= self.min_sample or bestErr < self.min_err:
            return Node(results=leaf(data))

        val_feat = []
        for feat in range(len(data[0]) - 1):
            val_feat = np.unique(data[:, feat])
            for val in val_feat:
                set_L, set_R = split_tree(data, feat, val)
                comb_L = combine(set_L[0], set_L[1])
                comb_R = combine(set_R[0], set_R[1])
                err_now = err_cnt(comb_L) + err_cnt(comb_R)
                if len(comb_L) < 2 or len(comb_R) < 2:
                    continue
                if err_now < bestErr:
                    bestErr = err_now
                    bestCreteria = (feat, val)
                    bestSets = (set_L, set_R)

        if bestErr > self.min_err:
            left = CART_regression(bestSets[0][0], bestSets[0][1], self.min_sample, self.min_err)
            right = CART_regression(bestSets[1][0], bestSets[1][1], self.min_sample, self.min_err)
            return Node(feature=bestCreteria[0], split_val=bestCreteria[1], left=left, right=right)
        else:
            return Node(results=leaf(data))

    def predicts(sample, tree):
        # 如果是叶子节点
        if tree.results is not None:
            return tree.results
        # 非叶子节点
        else:
            val_sample = sample[tree.feature]
            branch = None
            if val_sample < tree.split_val:
                branch = tree.left
            else:
                branch = tree.right
        return Node(sample, branch)

    def cal_err(Y_test, predicts):
        y_test = np.array(Y_test)
        pre_y = np.array(predicts)
        error = np.square(y_test - pre_y).sum() / len(Y_test)
        return error
