class Evaluation:
    pass


# 分类评价
# 准确率
class Accuracy(Evaluation):
    pass


# F1度量
class F1(Evaluation):
    pass


# P-R曲线，可视化
class PR(Evaluation):
    pass


# ROC曲线，可视化
class ROC(Evaluation):
    pass


# AUC
class AUC(Evaluation):
    pass


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
