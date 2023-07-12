class Model:
    pass


# 线性模型，线性回归->线性，逻辑回归->二分类
class LinearRegression(Model):
    pass


# 决策树模型（CART，C4.5）->分类
class DecisionTree(Model):
    pass


# 贝叶斯模型（贝叶斯分类器），朴素贝叶斯->分类
class NaiveBayes(Model):
    pass


# KNN模型，K最近邻算法->回归、分类
class KNN(Model):
    pass


# 支持向量机模型，分类
class SVM(Model):
    pass


# 随机森林模型，分类
class RandomForest(Model):
    pass


# 降维算法，降维
class DimensionalReduction(Model):
    pass


# XGBOOST，梯度增强算法->分类
class GradientBoosting(Model):
    pass


# CNN，梯度增强算法->分类
class CNN(Model):
    pass
