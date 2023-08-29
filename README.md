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
### 2. 逻辑回归算法
### 3. 决策树算法
### 4. 朴素贝叶斯算法
### 5. K最近邻算法
K最近邻算法（K-Nearest Neighbors，KNN）是一种基于实例的学习算法，用于分类和回归任务。KNN算法的原理可以概括为以下几个步骤：
    - 准备数据集：收集带有标签（类别或数值）的训练样本数据。
    - 计算距离：对于一个待预测的样本，计算它与训练样本之间的距离。常见的距离度量方法包括欧氏距离、曼哈顿距离、闵可夫斯基距离等。
    - 选择K个邻居：根据计算得到的距离，选择与待预测样本距离最近的K个训练样本作为邻居。
    - 确定类别（分类任务）或计算数值（回归任务）：对于分类任务，统计K个邻居中各个类别的出现频率，选择出现最多的类别作为预测结果。对于回归任务，取K个邻居数值特征的平均值作为预测结果。
    - 输出预测结果：返回预测结果（类别或数值）作为算法的输出。
KNN算法的优点包括简单易懂、无需训练过程、适用于多分类和回归问题。然而，KNN算法也有一些缺点，包括计算复杂度高、对特征缩放敏感、存储全部训练数据等。
在应用KNN算法时，通常需要根据具体问题选择合适的K值，并对数据进行适当的预处理操作，如特征缩放、标准化等，以提高算法性能和准确性。
### 6. 支持向量机算法
### 7. 随机森林算法
### 8. K均值聚类算法
K均值聚类（K-means clustering）是一种常用的无监督学习算法，用于将数据集分成K个不同的簇。该算法的原理可以概括为以下几个步骤：
    - 初始化聚类中心：随机选择K个初始聚类中心（也可以通过其他方式选择初始聚类中心），每个聚类中心是一个向量。
    - 分配样本到最近的聚类中心：对于每个样本，计算它与各个聚类中心之间的距离，并将其分配给距离最近的聚类中心所属的簇。
    - 更新聚类中心：对于每个簇，计算该簇中所有样本的均值，将均值作为新的聚类中心。
    - 重复前两个步骤，直到聚类中心不再发生变化或达到预定的迭代次数。
    - 输出结果：得到K个簇和它们对应的聚类中心，样本被分配到各个簇中。
K均值聚类的目标是最小化每个样本与所属簇聚类中心之间的距离之和（也称为簇内平方和，Sum of Squared Errors，SSE）。因此，该算法属于一种优化问题，通过迭代优化来逼近最优解。
K均值聚类的特点包括易于实现、计算效率高、扩展性强。然而，由于聚类结果受初始聚类中心的选择影响，可能会陷入局部最优解。为了克服这个问题，通常可以采用多次运行K均值聚类算法并选择最好的结果，或者使用其他启发式方法来选择初始聚类中心。
在应用K均值聚类算法时，需要选择合适的K值，并根据具体问题和数据集的特征进行适当的预处理，如数据标准化、处理离群值等，以提高聚类结果的质量和准确性。
### 9. 降维算法

### 10. 梯度增强算法

## 四、算法性能比较与分析

## 五、架构设计思路
> 描述了系统各个组成部分之间的关系、功能模块的划分以及它们之间的通信方式和交互方式
### （一）整体架构

### （二）后端架构
> 后端分为数据集，数据分割，模型、评估和工厂模式五个功能模块，每个模块均设置父类，分别为Dataset类，Splitter类，Model类、Evaluation类和Factory类。  
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
#### 5. 工厂模式模块
定义了一个 Factory 类和三个继承自 Factory 的子类：DataFactory、SplitterFactory 和 ModelFactory，以及一个独立的 EvaluationFactory 类。
- DataFactory 类
    - DataFactory 类通过 create_dataset 方法根据传入的数据集名称选择创建对应的数据集对象，目前支持的数据集有 iris、wine 和 heart disease。
- SplitterFactory 类
    - SplitterFactory 类通过 create_splitter 方法根据传入的分割器类型、特征矩阵 X、标签向量 y 和分割比例 percent 创建对应的数据集分割对象，目前支持的分割器有 bootstraping 和 holdout。
-ModelFactory 类
    -ModelFactory 类通过 create_model 方法根据传入的模型名称创建对应的模型对象，目前支持的模型有 KNN、K_means、Decision Tree、GradientBoosting、LR、SVM、LogisticRegression、NB 和 Random Forest。
-EvaluationFactory 类
    -EvaluationFactory 类通过 create_evaluation 方法根据传入的评估指标、真实标签 y_true、预测标签 y_pred、测试数据 x_test 和模型对象创建对应的评估对象，目前支持的评估指标有 auc、accuracy、distance、f1、fm、mse、pr、rmse、roc 和 rand。
### （三）前端架构
####  1. web.html
- <head>标签
设置文档的标题为“机器学习”，引入名为"web.css"的样式表文件用于设置页面的样式，名为"client.js"和"selection.js"的JavaScript脚本文件。
- <body>标签
<h1>定义一级标题为“机器学习模型评估”，使用<p align=”center”>使得文本居中对齐。
    - 在id为”selection-area”的模块，包含了一系列的选项：用于用户选择数据集、分割器、分割比例、机器学习模型和模型评估指标。这些选项使用了居中对齐的二级标题，并用<div>和class属性进行包裹和样式控制。
        - 用于选择数据集的radio按钮，class为"radio-container"，包含iris dataset,wine dataset和heart disease dataset三个选项。
        - 用于选择分割器的radio按钮，包含set-side method和bootstraping method两个选项。
        - 用于选择分割比例的radio按钮，包含10%和30%两个选项。
        - 用于选择机器学习模型的radio按钮，包含KNN,K_means,SVM,GradientBoosting,linear regression,logistic regression,navie bayes,decision tree和random forest九个选项。
        - 用于选择模型评估指标的radio按钮，包含Accuracy,Distance,AUC,F1,FM,MSE,PR,Rand,RMSE和ROC十个选项。
    - 在id为“action-area”的模块，创建了一个操作区域，包含了"运行"和"重置"两个按钮。点击"运行"按钮会调用send()函数，点击"重置"按钮会调用resetForm()函数。这些按钮使用<input>标签创建，type属性指定按钮类型，value属性设置按钮显示的文本内容。
    - 在id为“action-title”的模块，创建了一个带有"运行结果"标题的区域。
    - 在id为“result-area”的模块，创建了一个用于显示结果的空白区域。
整体实现了一个网页应用，用户可以通过选择选项来配置机器学习模型评估的各项参数，并点击"运行"按钮来获取相应的评估结果。页面中的样式表和脚本文件将用于控制网页的样式和实现交互功能。
        
####  2. web.css
web.css用于设置选择区域、运行区域和结果显示区域的样式。
- selection-area 是选择区域的样式，设置了高度为650px，背景色为#f2f2f2（浅灰色）。
- .radio-container 是一个容器，用于包裹选项按钮。设置了 Flex 布局，并使其内部元素居中对齐和垂直居中。
- input[type="radio"] 是选项按钮的样式，设置了右边距为5px。
- action-area 是运行区域的样式，设置了高度为50px，背景色为#ffffff（白色）。
- result-area 是结果显示区域的样式，设置了高度为300px，背景色为#ffffff（白色）。
- .evaluationOption, .modelOption, .datasetOption, .splitterOption 是用于设置不同选项的样式，都使用了 Inline-block 布局，并设置了右外边距为10px。
- .hidden 是一个类名，用于隐藏元素，设置了 display 属性为 none。
- h1 是标题的样式，设置了字体为"微软雅黑"，居中对齐。
这些样式通过 ID（以 # 开头）或类名（以 . 开头）来选取对应的元素，并为其设置相应的样式属性。通过这样的设置，可以使选择区域、运行区域和结果显示区域拥有特定的外观和布局。   
####  3. selection.js
selection.js用于实现交互的网页界面，包含了一些函数用于更新页面中的选项和元素显示状态，以及重置表单。
- updateOptions() 函数用于根据选择的数据集来更新显示的选项。根据选择的数据集（通过获取选中的 radio 按钮的值），设置不同选项的显示状态。例如，如果选择的数据集是 "iris"，则将距离选项、MSE 选项和 RMSE 选项隐藏起来；如果选择的数据集是 "heart"，则显示 AUC 选项、F1 选项、PR 选项和 ROC 选项等。
- updatepercent() 函数用于根据选择的划分方法来更新百分比选项的显示状态。根据选择的划分方法（通过获取选中的 radio 按钮的值），设置百分比选项的显示状态。例如，如果选择的划分方法是 "houldout"，则显示百分比选项；否则隐藏百分比选项。
- updatedata() 函数用于根据选择的模型来更新数据选项的显示状态。根据选择的模型（通过获取选中的 radio 按钮的值），设置不同数据选项的显示状态。例如，如果选择的模型是 "K_means"、"KNN" 或 "NB"，则显示 iris 选项；如果选择的模型是 "SVM" 或 "LogisticRegression"，则隐藏 wine 选项等。
- updatedata1() 函数用于根据选择的评估方法来更新数据选项的显示状态。根据选择的评估方法（通过获取选中的 radio 按钮的值），设置不同数据选项的显示状态。例如，如果选择的评估方法是 "accuracy"、"fm" 或 "rand"，则显示 iris 选项；如果选择的评估方法是 "auc"、"f1"、"pr" 或 "roc"，则隐藏 wine 选项等。
- updatesplitter() 函数用于根据选择的百分比划分方法来更新划分选项的显示状态。根据选择的百分比划分方法（通过获取选中的 radio 按钮的值），设置不同划分选项的显示状态。例如，如果选择的百分比划分方法是 "10%" 或 "30%"，则隐藏 CV 选项和 Bootstraping 选项等。
- resetForm() 函数用于重置表单，将所有 radio 按钮的选中状态取消，并刷新页面。
这些函数通常用于在用户与页面中的选项进行交互时，根据选择的结果动态更新页面的显示状态，以提供更好的用户体验。

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
