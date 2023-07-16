
import numpy as np
class NB:
    '''朴素贝叶斯分类器'''

    def __init__(self, data, label):
        self.data = data  # 训练数据
        self.label = label  # 训练数据类别
        self.n = len(data)  # 数据样本个数
        self.p = len(data[0])  # 属性个数
        self.pCondition, self.pClass = self.train()

    def Cal_class(self):
        '''计算训练样本的类别'''
        c = set()
        for i in self.label:
            c.add(i)
        return c

    def Cal_value(self, attr):
        '''计算属性attr的可能取值'''
        v = set()
        for i in self.data[:, attr]:
            v.add(i)
        return v

    def train(self):
        '''计算P(c)与P(x|c)'''
        str_type = type(np.array(['a'])[0])  # 用于后面属性数据类型的判断，此处为numpy.str类型
        int_type = type(np.array([1])[0])  # numpy.int类型
        float_type = type(np.array([1.1])[0])  # numpy.float类型
        c = self.Cal_class()
        pClass = {}  # 用于储存各类的概率P（c）
        numClass = {}  # 用于储存各类的个数
        pCondition = [{} for i in range(self.p)]  # 列表中每个字典用于储存对应属性的类条件概率；若为连续性，储存μ与σ
        for i in range(self.n):
            if self.label[i] not in pClass:
                pClass[self.label[i]] = 1 / (self.n + len(c))  # 分子加1，为拉普拉斯修正
                numClass[self.label[i]] = 0
            pClass[self.label[i]] += 1 / (self.n + len(c))  # 统计各类出现的概率
            numClass[self.label[i]] += 1  # 统计各类的数值
        for j in range(self.p):
            '''逐个属性计算'''
            if type(self.data[0][j]) == str_type:
                '''计算离散型属性的类条件概率'''
                v = self.Cal_value(attr=j)
                for i in range(self.n):
                    if self.label[i] not in pCondition[j]:
                        init_num = [1 / (pClass[self.label[i]] + len(v)) for i in range(len(v))]  # 初始数据，并采用拉普拉斯修正
                        pCondition[j][self.label[i]] = dict(zip(v, init_num))  # 用于统计label[i]类下的j属性的条件概率
                    pCondition[j][self.label[i]][self.data[i][j]] += 1 / (pClass[self.label[i]] + len(v))
            elif type(self.data[0][j]) == float_type or int_type:
                '''计算连续型属性的类条件概率'''
                data_class = {}  # 储存该属性中各类对应的数据
                for i in range(self.n):
                    if self.label[i] not in data_class:
                        data_class[self.label[i]] = [self.data[i][j]]
                    else:
                        data_class[self.label[i]].append(self.data[i][j])
                for key in data_class.keys():
                    miu, sigma = np.mean(data_class[key]), np.var(data_class[key])
                    pCondition[j][key] = {'miu': miu, 'sigma': sigma}
        return pCondition, pClass

    def Cal_p(self, attr, x, c):
        '''计算属性attr中值为x的c类条件概率：P(x|c)
        attr:属性，值为0~p-1
        x:值，为连续的数或离散的值
        c:类'''
        if 'miu' and 'sigma' in self.pCondition[attr][c]:
            '''判断是否为连续型属性，此时是连续型属性'''
            miu = self.pCondition[attr][c]['miu']
            sigma = self.pCondition[attr][c]['sigma']
            p = np.exp(-(x - miu) ** 2 / (2 * sigma)) / np.sqrt(2 * np.pi * sigma)
            return p
        else:
            p = self.pCondition[attr][c][x]
            return p

    def predict(self, x):
        '''根据一组数据x预测其属于哪一类
        x:长度为p的列表或array类型'''
        p = {}  # 储存属于各类的概率
        for c in self.pClass.keys():
            pc = np.log(self.pClass[c])
            for i in range(self.p):
                pc += np.log(self.Cal_p(attr=i, x=x[i], c=c))

            p[c] = pc
        maxp = max(p.values())  # 选取最大概率
        for key, value in p.items():
            if value == maxp:
                return key

    def test(self, testData, testLabel):
        '''利用测试集测试模型准确性'''
        n = len(testData)
        correct = 0  # 统计正确的个数
        for i in range(n):
            if self.predict(testData[i]) == testLabel[i]:
                correct += 1
        return correct / n

