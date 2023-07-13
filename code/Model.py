class Model:
    pass


# 线性模型，线性回归->线性，逻辑回归->二分类
class LinearRegression(Model):
    pass


# 决策树模型（ID3）->分类
class DecisionTree(Model):
    def compute_entropy(dataset):
    # 求总样本数
    num_of_examples = len(dataset)
    labelCnt = {}
    # 遍历整个样本集合
    for example in dataset:
        # 当前样本的标签值是该列表的最后一个元素
        currentLabel = example[-1]
        # 统计每个标签各出现了几次
        if currentLabel not in labelCnt.keys():
            labelCnt[currentLabel] = 0
        labelCnt[currentLabel] += 1
    entropy = 0.0
    # 对于原样本集，labelCounts = {'no': 6, 'yes': 9}
    # 对应的初始shannonEnt = (-6/15 * log(6/15)) + (- 9/15 * log(9/15))
    for key in labelCnt:
        p = labelCnt[key] / num_of_examples
        entropy -= p * log(p, 2)
    return entropy


    # 提取子集合
    # 功能：从dataSet中先找到所有第axis个标签值 = value的样本
    # 然后将这些样本删去第axis个标签值，再全部提取出来成为一个新的样本集
    def create_sub_dataset(dataset, index, value):
        sub_dataset = []
        for example in dataset:
            current_list = []
            if example[index] == value:
                current_list = example[:index]
                current_list.extend(example[index + 1:])
                sub_dataset.append(current_list)
        return sub_dataset
    
    
    def choose_best_feature(dataset):
        num_of_features = len(dataset[0]) - 1
        # 计算当前数据集的信息熵
        current_entropy = compute_entropy(dataset)
        # 初始化信息增益率
        best_information_gain_ratio = 0.0
        # 初始化最佳特征的下标为-1
        index_of_best_feature = -1
        # 通过下标遍历整个特征列表
        for i in range(num_of_features):
            # 构造所有样本在当前特征的取值的列表
            values_of_current_feature = [example[i] for example in dataset]
            unique_values = set(values_of_current_feature)
            # 初始化新的信息熵
            new_entropy = 0.0
            # 初始化分离信息
            split_info = 0.0
            for value in unique_values:
                sub_dataset = create_sub_dataset(dataset, i, value)
                p = len(sub_dataset) / len(dataset)
                # 计算使用该特征进行样本划分后的新信息熵
                new_entropy += p * compute_entropy(sub_dataset)
                # 计算分离信息
                split_info -= p * log(p, 2)
            # 计算信息增益
            # information_gain = current_entropy - new_entropy
            # 计算信息增益率（Gain_Ratio = Gain / Split_Info）
            information_gain_ratio = (current_entropy - new_entropy) / split_info
            # 求出最大的信息增益及对应的特征下标
            if information_gain_ratio > best_information_gain_ratio:
                best_information_gain_ratio = information_gain_ratio
                index_of_best_feature = i
        # 这里返回的是特征的下标
        return index_of_best_feature
    
    
    # 返回具有最多样本数的那个标签的值（'yes' or 'no'）
    def find_label(classList):
        # 初始化统计各标签次数的字典
        # 键为各标签，对应的值为标签出现的次数
        labelCnt = {}
        for key in classList:
            if key not in labelCnt.keys():
                labelCnt[key] = 0
            labelCnt[key] += 1
        # 将classCount按值降序排列
        # 例如：sorted_labelCnt = {'yes': 9, 'no': 6}
        sorted_labelCnt = sorted(labelCnt.items(), key=lambda a: a[1], reverse=True)
        # 下面这种写法有问题
        # sortedClassCount = sorted(labelCnt.iteritems(), key=operator.itemgetter(1), reverse=True)
        # 取sorted_labelCnt中第一个元素中的第一个值，即为所求
        return sorted_labelCnt[0][0]
    
    
    def create_decision_tree(dataset, features):
        # 求出训练集所有样本的标签
        # 对于初始数据集，其label_list = ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
        label_list = [example[-1] for example in dataset]
        # 先写两个递归结束的情况：
        # 若当前集合的所有样本标签相等（即样本已被分“纯”）
        # 则直接返回该标签值作为一个叶子节点
        if label_list.count(label_list[0]) == len(label_list):
            return label_list[0]
        # 若训练集的所有特征都被使用完毕，当前无可用特征，但样本仍未被分“纯”
        # 则返回所含样本最多的标签作为结果
        if len(dataset[0]) == 1:
            return find_label(label_list)
        # 下面是正式建树的过程
        # 选取进行分支的最佳特征的下标
        index_of_best_feature = choose_best_feature(dataset)
        # 得到最佳特征
        best_feature = features[index_of_best_feature]
        # 初始化决策树
        decision_tree = {best_feature: {}}
        # 使用过当前最佳特征后将其删去
        del (features[index_of_best_feature])
        # 取出各样本在当前最佳特征上的取值列表
        values_of_best_feature = [example[index_of_best_feature] for example in dataset]
        # 用set()构造当前最佳特征取值的不重复集合
        unique_values = set(values_of_best_feature)
        # 对于uniqueVals中的每一个取值
        for value in unique_values:
            # 子特征 = 当前特征（因为刚才已经删去了用过的特征）
            sub_features = features[:]
            # 递归调用create_decision_tree去生成新节点
            decision_tree[best_feature][value] = create_decision_tree(
                create_sub_dataset(dataset, index_of_best_feature, value), sub_features)
        return decision_tree



# 贝叶斯模型（贝叶斯分类器），朴素贝叶斯->分类
class NaiveBayes(Model):
    pass


# KNN模型，K最近邻算法->回归、分类
class KNN(Model):
    #计算每个点之间的距离
    def distancecount(instance1,instance2,length):
        distance=0
        for i in range(length):
            distance+=pow((instance1[i]-instance2[i]),2)
        return math.sqrt(distance)
    #获取k个邻居
    def kneighbors(train_data,test_data,k):
        distance1=[]
        length=len(test_data)-1
        for i in range(len(train_data)):
            dis = train_data.distancecount(test_data, train_data[i], length)
            distance1.append(train_data[i],dis)
        distance1.sort(key=operator.itemgetter(1))
        kneightbors=[]
        for i in range(k):
            kneightbors.append(distance1[i][0])
            return kneightbors
    #获取最多类别的类
    def most(kneightbors):
        class1=[]
        for i in range(len(kneightbors)):
            most=kneightbors[i][-1]
            if most in class1:
                class1[most]+=1
            else:
                class1[most]=1
        sortclass=sorted(class1.items(),key=operator.itemgetter(1),reverse=True)
        return sortclass

# 支持向量机模型，分类
class SVM(Model):
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel

    # 参数初始化
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        self.alpha = np.ones(self.m)
        self.computer_product_matrix()  # 为了加快训练速度创建一个内积矩阵
        # 松弛变量
        self.C = 1.0
        # 将Ei保存在一个列表里
        self.create_E()

    # KKT条件判断
    def judge_KKT(self, i):
        y_g = self.function_g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # 计算内积矩阵，如果数据量较大，可以使用系数矩阵
    def computer_product_matrix(self):
        self.product_matrix = np.zeros((self.m, self.m)).astype(np.float)
        for i in range(self.m):
            for j in range(self.m):
                if self.product_matrix[i][j] == 0.0:
                    self.product_matrix[i][j] = self.product_matrix[j][i] = self.kernel(self.X[i], self.X[j])

    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return np.dot(x1, x2)
        elif self._kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** 2
        return 0

    # 将Ei保存在一个列表里
    def create_E(self):
        self.E = (np.dot((self.alpha * self.Y), self.product_matrix) + self.b) - self.Y

    # 预测函数g(x)
    def function_g(self, i):
        return self.b + np.dot((self.alpha * self.Y), self.product_matrix[i])

    # 选择变量
    def select_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self.judge_KKT(i):
                continue
            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = np.argmin(self.E)
            else:
                j = np.argmax(self.E)
            return i, j

    # 剪切
    def clip_alpha(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    # 训练函数，使用SMO算法
    def Train(self, features, labels):
        self.init_args(features, labels)
        # SMO算法训练
        for t in range(self.max_iter):
            i1, i2 = self.select_alpha()

            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(self.X[i1], self.X[i2])
            if eta <= 0:
                # print('eta <= 0')
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            alpha2_new = self.clip_alpha(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.create_E()

    # 预测
    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1

    # 简单评估
    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)


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
