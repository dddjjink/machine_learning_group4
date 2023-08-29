# machine_learning_group4

## 一、组员与分工
### （一）小组成员
- 李青琪 2021212023018
- 柴菲儿 2021212023004
- 张晨宇 2021212023007
- 吴  倩 2021212023028
### （二）小组分工
1. 后端
    1. 数据集部分
        - 数据集选取：张晨宇
        - 数据集收集：吴倩
        - 数据加载代码实现、代码整改：李青琪
    2. 数据分割部分
        - 留出法：吴倩
        - 交叉验证法：张晨宇
        - 自助法、代码整改：李青琪
    3. 模型训练部分
        - 决策树模型、随机森林模型：吴倩
        - 支持向量机模型、GBDT模型、代码整改：李青琪
        - 朴素贝叶斯模型、线性回归模型、逻辑回归模型、降维模型：柴菲儿
        - KNN模型、K均值聚类算法：张晨宇
    4. 模型评估部分
        - MSE、RMSE、Distance、代码整改：李青琪
        - FM、Rand：吴倩
        - Accuracy、F1、PR、AUC、ROC：柴菲儿
3. 前端：张晨宇、柴菲儿
4. 前后端通信：吴倩
5. 优化改进：张晨宇、柴菲儿、吴倩、李青琪

## 二、机器学习总体流程介绍
> 数据集载⼊、分割、模型训练、测试、评估等

## 三、算法原理介绍
### 1. 线性回归算法
线性回归模型用于建模和预测连续数值输出。它假设输入特征与输出之间存在线性关系。线性回归模型通过最小化预测值与观测值之间的残差平方和来拟合一个线性方程,即f(xi) = wxi + b，使得f(xi) ≈ yi。在训练过程中，模型通过寻找最优的系数来使得预测值与实际观测值尽可能接近，从而得到一个线性函数来进行预测。
求解最优系数的方式有两种：第一种是最小二乘法，最小二乘法就是试图找到一条直线，使得所有样本数据点到达直线的欧氏距离最小，总距离是所有数据点的垂直距离的平方和，其思想是通过最小化这个平方误差或距离来拟合模型；第二种是梯度下降，梯度下降核心内容是对自变量进行不断的更新（针对w和b求偏导），使得目标函数不断逼近最小值的过程。
### 2. 逻辑回归算法
逻辑回归是一种用于分类问题的统计学习方法。在逻辑回归中，我们要预测的是一个定性变量，比如y=0或1，表示两个类别。逻辑回归通过建立一个代价函数，并通过优化方法迭代求解出最优的模型参数来进行分类。 不过需要注意的是，虽然逻辑回归名字中带有“回归”，但它实际上是一种分类方法，主要用于两分类问题。
具体而言，逻辑回归的原理是先将样本的特征值线性组合，然后通过一个逻辑函数（也称为sigmoid函数）将线性组合的结果映射到一个概率值，再根据阈值将概率值转化为类别标签。逻辑函数的形式是1/(1+e^(-z))，其中z是线性组合的结果。概率值越接近1，表示属于某个特定类别的概率越大；概率值越接近0，表示属于另一个类别的概率越大。
逻辑回归的优点是计算简单、易于理解和解释，适用于处理二分类问题，并且对于高维数据也能较好地处理。然而，逻辑回归也有一些缺点，比如对于非线性关系的数据拟合能力较弱，容易受到异常值的影响。此外，逻辑回归是线性模型，无法处理特征之间的复杂交互关系。
### 3. 决策树算法
### 4. 朴素贝叶斯算法
朴素贝叶斯模型基于贝叶斯定理和特征条件独立假设。朴素贝叶斯模型是一种基于贝叶斯定理的概率模型，常用于文本分类、垃圾邮件过滤、情感分析等任务。其算法原理如下：假设我们有一个训练集，其中包含一组已分类的样本数据。首先，需要统计每个类别的先验概率，即每个类别在训练集中出现的频率。对于每个特征，在每个类别下计算其条件概率。假设输入样本有n个特征，我们需要计算每个类别下的联合概率，即 P(类别 | 特征1, 特征2, …, 特征n)。根据贝叶斯定理，可以计算出后验概率 P(类别 | 特征1, 特征2, …, 特征n) = P(特征1, 特征2, …, 特征n | 类别) * P(类别) / P(特征1, 特征2, …, 特征n)。对于给定的输入样本，计算其属于每个类别的后验概率，并选择具有最高后验概率的类别作为预测结果。
在朴素贝叶斯模型中，假设所有特征之间是相互独立的，这是“朴素”的原因。虽然这个假设在现实中很少成立，但朴素贝叶斯模型仍然可以取得不错的分类效果，且计算效率高。
### 5. K最近邻算法
### 6. 支持向量机算法
### 7. 随机森林算法
### 8. K均值聚类算法
### 9. 降维算法
降维模型用于减少高维数据集的维度，并保留最重要的特征。降维的目的是降低数据存储和计算复杂度，并提高模型的效率和泛化能力。常用的降维方法包括主成分分析（PCA）和线性判别分析（LDA）。在本项目中用的降维方法是主成分分析，主成分分析是一种无监督的降维技术，通过将原始数据投影到新的低维子空间来实现降维。其算法原理如下：先计算原始数据的协方差矩阵，再对协方差矩阵进行特征值分解，得到特征值和对应的特征向量。然后根据特征值的大小排序特征向量，选取前k个特征向量作为主成分（k为降维后的目标维度）。最后将原始数据投影到所选的主成分上，得到降维后的数据。
PCA的关键思想是通过寻找协方差矩阵的主要特征向量，找到能够最好地解释原始数据变化的低维表示。通常情况下，选择的前几个主成分可以保留大部分的数据方差，从而更好地表示数据。
### 10. 梯度增强算法

## 四、算法性能比较与分析

## 五、架构设计思路
> 描述了系统各个组成部分之间的关系、功能模块的划分以及它们之间的通信方式和交互方式
### （一）整体架构

### （二）后端架构
> 后端分为数据集，数据分割，模型和评估四个功能模块，每个模块均设置父类，分别为Dataset类，Splitter类，Model类和Evaluation类。  
#### 1. 数据集模块
Dataset类包含__init__函数，load函数和data_target_split函数：
- __init__函数，即初始化函数。该函数对路径、数据集、特征和标签进行初始化。
- load函数，即数据载入函数。该函数根据路径对数据集进行赋值。
- data_target_split函数，即数据集分割函数。该函数对特征和标签进行赋值。

Dataset类有三个子类，分别为IrisDataset类，WineQualityDataset类和HeartDiseaseDataSet类。每个子类均包含__init__函数和data_target函数且每个子类函数的内容架构相同：
- __init__函数，即初始化函数。该函数对路径进行初始化。
- data_target函数，即数据集分割函数。该函数首先调用父类的load函数进行数据集的加载，然后根据列名划分特征和标签。
#### 2. 数据分割模块
Splitter类包含__init__函数，即初始化函数，该函数对特征和标签进行初始化。

Splitter类有两个子类，分别为HoldOut类和BootStrapping类。每个子类均包含__init__函数，split函数：
- HoldOut类
    - __init__函数，即初始化函数。该函数对特征、标签、数据分割比例和随机状态进行初始化。
    - split函数，即数据分割函数。该函数首先根据随机状态设置随机种子，再获取样本数量和测试集大小，然后随机选择测试集的索引，最后依据索引构建训练集和测试集。
- BootStrapping类
    - __init__函数，即初始化函数。该函数对特征和标签进行初始化。
    - split函数，即数据分割函数。该函数首先进行m次放回抽样，得到训练集的序号，然后将剩下的序号记为测试集序号，最后产生训练/测试集。
#### 3. 模型模块
Model类有八个子类，分别为DecisionTree类，SVM类，GBDT类，NB类，LinearRegression类，LogisticRegression类，KNN类和KMeans类。除了这八个子类，该模块还有一个继承自DecisionTree类的RandomForest类和一个PCA类。  
- DecisionTree类，SVM类，GBDT类，NB类，LinearRegression类，LogisticRegression类，KNN类，KMeans类
    - 主要函数为fit函数和predict函数，其他函数根据这两个函数所需来写。
    - fit函数，即训练函数。该函数负责模型的训练。
    - predict函数，即预测函数。该函数负责模型的预测。
- RandomForest类
    - 继承自DecisionTree类，主要函数为fit函数和predict函数，同时会用到DecisionTree类来构建所需的决策树。
- PCA类
    - LinearRegression类，LogisticRegression类，KNN类和KMeans类中调用了该类的transform函数。
    - 当数据的维度大于4时，调用该函数对数据进行降维处理。
- 本项目中的模型与适用的问题以及适用数据集的关系【表述为模型（适用的问题）：适用的数据集】
    - DecisionTree（回归问题）：WineQualityDataset，HeartDiseaseDataSet
    - SVM（二分类问题）：HeartDiseaseDataSet
    - GBDT（分类问题）：WineQualityDataset，HeartDiseaseDataSet
    - NB（分类问题）：IrisDataset，WineQualityDataset，HeartDiseaseDataSet
    - LinearRegression（回归问题）：WineQualityDataset，HeartDiseaseDataSet
    - LogisticRegression（二分类问题）：HeartDiseaseDataSet
    - KNN（分类问题、回归问题）：IrisDataset，WineQualityDataset，HeartDiseaseDataSet
    - KMeans（聚类问题）：IrisDataset，WineQualityDataset，HeartDiseaseDataSet
    - RandomForest（分类问题、回归问题）：WineQualityDataset，HeartDiseaseDataSet
#### 4. 评估模块
Evaluation类包含__init__函数，即初始化函数，该函数依据传入的参数对真实值和预测值进行初始化。  

Evaluation类有十个子类，分别为Accuracy类，F1类，PR类，AUC类，ROC类，FM类，Rand类，MSE类，RMSE类和Distance类。每个子类都使用__call__方法，在评估时，只需要将真实值和预测值传给实例对象，然后调用该实例对象。每个子类都会返回信息到前端，详细内容见前端和前后端通信模块。
- Accuracy类，F1类，PR类，AUC类，ROC类，FM类，Rand类，MSE类，RMSE类和Distance类
    - Accuracy类
        - 该类的__call__中首先调用accuracy_cal函数，然后调用并返回evaluate函数。
        - accuracy_cal函数统计真正例、假正例、真反例、假反例的数目。
        - evaluate函数计算并返回准确率。
    - F1类
        - 该类的__call__中首先调用f1_score函数，然后调用并返回evaluate函数。
        - f1_score函数统计真正例、真反例、假反例的数目。
        - evaluate函数计算并返回F1度量。
    - PR类
        - 该类的__call__中首先调用plot函数，然后返回'curve'字符串。
        - plot函数需要调用precision_recall_curve函数来获取精确率和召回率，并根据精确率和召回率来作出曲线。
    - AUC类
        - 该类调用了ROC类
        - 该类的__call__中首先调用auc函数，然后调用并返回evaluate函数。
        - auc函数用到DecisionTree类来构建所需的ROC实例对象，然后调用roc_curve函数来获取真正例率和假正例率。
        - evaluate函数首先对假正例率列表进行排序，然后根据排序索引对真正例率列表进行排序，最后计算并返回排序后的真正例率和假正例率形成的曲线下的面积。
    - ROC类
        - 该类的__call__中首先调用plot函数，然后返回'curve'字符串。
        - plot函数需要调用roc_curve函数来获取真正例率和假正例率，并根据真正例率和假正例率来作出曲线。
    - FM类
        - 该类的__call__调用并返回compute_fm_index函数。
        - compute_fm_index函数需要调用compute_confusion_matrix函数来得到混淆矩阵，再根据根据混淆矩阵的真正例、假正例、真反例、假反例计算精确率和召回率，最后根据精确率和召回率计算并返回FM指数。
    - Rand类
        - 该类的__call__调用并返回compute_fm_index函数。
        - compute_fm_index函数需要调用compute_confusion_matrix函数来得到混淆矩阵，再根据根据混淆矩阵的真正例、假正例、真反例、假反例计算并返回Rand指数。
    - MSE类
        - 该类的__call__调用并返回loss函数。
        - loss函数依据公式计算并返回真实值与预测值的均方误差。
    - RMSE类
        - 该类的__call__调用并返回loss函数。
        - loss函数依据公式计算并返回真实值与预测值的均方根误差。
    - Distance类
        - 该类的__call__调用并返回minkowski_distance函数。
        - minkowski_distance函数依据公式计算并返回真实值与预测值的闵可夫斯基距离。
- 本项目中的评估方法与适用的问题以及适用数据集的关系【表述为评估方法（适用的问题）：适用的数据集】
    - Accuracy（分类问题）：IrisDataset，WineQualityDataset，HeartDiseaseDataSet
    - F1（分类问题）：HeartDiseaseDataSet
    - PR（分类问题）：HeartDiseaseDataSet
    - AUC（分类问题）：HeartDiseaseDataSet
    - ROC（分类问题）：HeartDiseaseDataSet
    - FM（聚类问题）：IrisDataset，WineQualityDataset，HeartDiseaseDataSet
    - Rand（聚类问题）：IrisDataset，WineQualityDataset，HeartDiseaseDataSet
    - MSE（回归问题）：WineQualityDataset，HeartDiseaseDataSet
    - RMSE（回归问题）：WineQualityDataset，HeartDiseaseDataSet
    - Distance（聚类问题）：WineQualityDataset，HeartDiseaseDataSet
### （三）前端架构

### （四）前后端连接架构

## 六、前后端代码仓库链接
https://github.com/stupid-vegetable-bird/machine_learning_group4.git

## 七、代码运行方式说明
运行code文件夹中的main.py文件，即下图中红框标记的文件  
![image](https://github.com/stupid-vegetable-bird/machine_learning_group4/assets/97822083/4fd61545-fa8e-47f3-922b-6c8af0d69ee0)

## 八、代码运行截图
![image](https://github.com/stupid-vegetable-bird/machine_learning_group4/assets/97822083/6a99b913-e810-4fcb-803c-154172db77cd)  
运行后的初始界面  

![image](https://github.com/stupid-vegetable-bird/machine_learning_group4/assets/97822083/38abdd5f-d904-49b6-bfaf-d9636eef7336)  
数据集、分割器。分割比例、模型、评估指标的选择及文字形式的运行结果  

![image](https://github.com/stupid-vegetable-bird/machine_learning_group4/assets/97822083/faf3f6a4-e90c-43f8-967c-412b173e39cd)  
图片形式的运行结果显示

## 九、每位组员的收获与感悟
### 1. 李青琪
经过小学期和暑假的学习，学到了手写机器学习模型代码、搭建前端和实现前后端通信的相关知识并加以实践，对项目开发以及团队合作开发有了进一步的了解与体会，对机器学习整个流程也有了更深入的认识。要手写机器学习模型代码就要去学习模型背后的算法的逻辑、了解并掌握相关的知识，在本次实践当中，如何参透算法的数学逻辑与内容并将其以代码的形式来更好地实现是我重点面对的问题。除此之外，在后端部分对代码进行整改也让我明白了良好的代码风格的重要性，以及在项目创建过程中如何与团队成员携手遵循共同且良好的代码风格。在前端和前后端通信部分，对websocket有了初步的学习与简单的实践，学到了前后端通信的知识与前后端通信的搭建。
### 2. 柴菲儿
### 3. 张晨宇
### 4. 吴  倩

## 十、其他重要的内容
