import numpy as np
import Evaluation
def compute_confusion_matrix(labels_true, labels_pred):
    n_samples = len(labels_true)
    confusion_matrix = np.zeros((2, 2))
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if labels_true[i] == labels_true[j] and labels_pred[i] == labels_pred[j]:
                confusion_matrix[0, 0] += 1  # true positive
            elif labels_true[i] != labels_true[j] and labels_pred[i] != labels_pred[j]:
                confusion_matrix[1, 1] += 1  # true negative
            elif labels_true[i] == labels_true[j] and labels_pred[i] != labels_pred[j]:
                confusion_matrix[0, 1] += 1  # false negative
            else:
                confusion_matrix[1, 0] += 1  # false positive
    
    return confusion_matrix

def compute_fm_index(labels_true, labels_pred):
    cm = compute_confusion_matrix(labels_true, labels_pred)
    tp = cm[0, 0]
    fp = cm[1, 0]
    fn = cm[0, 1]
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    fm_index = np.sqrt(precision * recall)
    
    return fm_index

#上述代码中，compute_confusion_matrix函数用于计算混淆矩阵，该矩阵用于计算FM指数和Rand指数。
#compute_fm_index函数通过混淆矩阵计算FM指数，公式为：FM = sqrt(precision * recall)，
#其中precision为精确率，recall为召回率。compute_rand_index函数通过混淆矩阵计算Rand指数，
#公式为：Rand = (TP + TN) / (TP + FP + FN + TN)，其中TP表示真正例数量，TN表示真反例数量，FP表示假正例数量，FN表示假反例数量。

# 生成随机的标签数据
labels_true = np.array([0, 0, 1, 1, 1])
labels_pred = np.array([0, 0, 1, 1, 0])

fm_index = compute_fm_index(labels_true, labels_pred)
#rand_index = compute_rand_index(labels_true, labels_pred)

print("FM指数:", fm_index)
#print("Rand指数:", rand_index)
#请注意，在使用这些指标时，需要提供真实的类别标签（labels_true）和聚类算法预测的类别标签（labels_pred）。
#这里只提供了一个简单的示例，你可以根据实际需求对代码进行修改和扩展。
