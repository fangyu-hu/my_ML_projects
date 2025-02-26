import matplotlib.pyplot as plt
import numpy as np
# 用于创建颜色映射
from matplotlib.colors import ListedColormap
# make_circles: 生成“圆形”数据集
# make_classification: 生成“线性可分”数据集
# make_moons: 生成“月牙形”数据集
from sklearn.datasets import make_circles, make_classification, make_moons
# 二次判别分析
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# AdaBoostClassifier: AdaBoost分类器
# RandomForestClassifier: 随机森林分类器
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# 高斯过程分类器
from sklearn.gaussian_process import GaussianProcessClassifier
# 径向基函数
from sklearn.gaussian_process.kernels import RBF
# 用于绘制分类器的决策边界
from sklearn.inspection import DecisionBoundaryDisplay
# 用于划分训练集和测试集
from sklearn.model_selection import train_test_split
# 高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
# 最近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 多层感知器分类器
from sklearn.neural_network import MLPClassifier
# 创建包含多个步骤的管道
from sklearn.pipeline import make_pipeline
# 用于标准化特征
from sklearn.preprocessing import StandardScaler
# 支持向量机分类器
from sklearn.svm import SVC
# 决策树分类器
from sklearn.tree import DecisionTreeClassifier

# 包含字符串的列表，每个字符串是一个分类器的名称
# 这些名称用于标识和参考不同的分类器
# Nearest Neighbors: 最近邻分类器
# Linear SVM: 线性支持向量机分类器
# RBF SVM: 径向基函数核的支持向量机分类器
# Gaussian Process: 高斯过程分类器
# Decision Tree: 决策树分类器
# Random Forest: 随机森林分类器
# Neural Net: 多层感知器分类器
# AdaBoost: AdaBoost分类器
# Naive Bayes: 高斯朴素贝叶斯分类器
# QDA: 二次判别分析
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]
# 包含不同分类器实例的列表
# 每个分类器都用特定的参数初始化
classifiers = [
    # 最近邻分类器
    # 使用最近的3个邻居进行分类
    KNeighborsClassifier(3),
    # 线性支持向量机分类器
    # kernel="linear"使用线性核函数，C=0.025是正则化参数，random_state=42确保结果的可重现
    SVC(kernel="linear", C=0.025, random_state=42),
    # 径向基函数（RBF）核的SVM分类器。gamma=2 控制核函数的宽度，C=1正则化参数
    SVC(gamma=2, C=1, random_state=42),
    # 高斯过程分类器
    # 1.0 * RBF(1.0)定义协方差函数的参数，使用径向基函数作为协方差函数。random_state=42确保结果的可重现
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    # 决策树分类器
    # max_depth=5限制树的最大深度，random_state=42确保结果的可重现
    DecisionTreeClassifier(max_depth=5, random_state=42),
    # 随机森林分类器
    # max_depth=5定义树的最大深度，n_estimators=10 森林中树的数量，max_features=1限制每个决策树使用的特征数量。random_state=42确保结果的可重现
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    # 多层感知器分类器
    # alpha=1是L2惩罚（正则化项）的参数。max_iter=1000最大迭代次数。random_state=42确保结果的可重现
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    # AdaBoost分类器，random_state=42确保结果的可重现
    AdaBoostClassifier(random_state=42),
    # 高斯朴素贝叶斯分类器
    GaussianNB(),
    # 二次判别分析
    QuadraticDiscriminantAnalysis(),
]
# 生成一个用于分类的合成数据集
# n_features=2: 数据集有2个特征
# n_redundant=0: 没有冗余特征
# n_informative=2: 两个特征都是信息性的，即对分类任务有帮助
# random_state=1: 确保每次生成的数据都是一样的
# n_clusters_per_class=1: 每个类别生成的聚类数为1，每个类别的数据将围绕一个中心点聚集
# X: 特征集，y: 标签集
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
# 创建一个NumPy的随机数生成器实例，种子为2
rng = np.random.RandomState(2)
# 对之前生成的数据集X进行修改
# rng.uniform(size=X.shape)生成与X形状相同的随机数数组，随机数在[0, 1)区间内均匀分布
# 然后，这个数组乘以2后加到X上，对数据集加入了一定的噪声
X += 2 * rng.uniform(size=X.shape)
# 创建一个元组，包含处理过噪声的特征集X和对应的标签y
# 代表一个线性可分的数据集
linearly_separable = (X, y)
# 包含了三个数据集，用于后续的分类实验
# make_moons(noise=0.3, random_state=0): 生成“月牙形”数据集，noise=0.3为加入的噪声量
# make_circles(...): 生成“圆形”数据集，noise=0.2为噪声量，factor=0.5定义两个圆的大小关系，内圆的半径是外圆半径的一半
# linearly_separable: 前面定义的线性可分数据集
datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]
# 创建一个matplotlib图形对象，宽27英寸、高9英寸
figure = plt.figure(figsize=(27, 9))
# 用于后续在绘图时追踪子图的位置
i = 1
# iterate over datasets
# 使用enumerate函数对datasets列表进行迭代
# enumerate返回每个元素的索引（ds_cnt）和元素值（ds）
# ds是一个包含特征和标签的元组
for ds_cnt, ds in enumerate(datasets):
    # 将元组ds解包为特征集X和标签集y
    X, y = ds
    # 使用train_test_split函数将数据集划分为训练集和测试集
    # test_size=0.4表示40%的数据用于测试集，random_state=42保证划分的一致性
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    # 计算数据集中第一个特征（X[:, 0]）的最小值和最大值，并分别减去0.5和加上0.5
    # 为了在绘制图形时留出一定的边界空间
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    # 对第二个特征（X[:, 1]）进行操作
    # 为了在绘制图形时在y轴方向上留出边界空间
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # just plot the dataset first
    # 设置一个颜色映射变量cm，使用matplotlib的内置颜色映射RdBu（红蓝）
    # RdBu是一种双色调的映射，用于表示从一个极端到另一个极端的数据
    cm = plt.cm.RdBu
    # 创建一个新的颜色映射cm_bright，由—红色（#FF0000）和蓝色（#0000FF）构成的
    # 用于在散点图中区分不同的类别
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    # 创建一个子图ax
    # 函数plt.subplot的参数len(datasets)是行数，len(classifiers) + 1是列数，i是当前子图的索引
    # 每个数据集的每个分类器都会有一个子图，加上一个额外的子图用于显示输入数据
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # 用于只在第一个数据集的第一个子图上设置标题“Input data”
    # ds_cnt是数据集的索引，当其为0时，表示当前处理的是第一个数据集
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    # 使用scatter方法在子图ax上绘制训练数据点
    # X_train[:, 0]和X_train[:, 1]分别是训练数据的第一个特征和第二个特征，用于确定每个点的位置
    # c=y_train指定每个点的颜色，颜色基于训练标签y_train
    # cmap=cm_bright指定用于颜色映射的colormap，这里使用的是之前定义的红蓝色映射
    # edgecolors="k"给每个点加了黑色的边框
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # 用于绘制测试数据点
    # alpha=0.6设置点的透明度，区分训练点和测试点（测试点更透明）
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    # 设置子图的x轴和y轴的显示范围
    # x_min, x_max, y_min, y_max是之前计算的特征的最小和最大值，加上了一定的边界空间
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # 移除子图的x轴和y轴的刻度
    ax.set_xticks(())
    ax.set_yticks(())
    # 将子图索引i增加1，为下一个子图的创建做准备
    i += 1
    # iterate over classifiers
    # 使用zip函数将之前定义的分类器名称列表names和分类器实例列表classifiers合并，然后迭代它们
    # name和clf分别代表当前分类器的名称和实例
    for name, clf in zip(names, classifiers):
        # 创建一个新的子图ax
        # 函数plt.subplot的参数len(datasets)是行数，len(classifiers) + 1是列数，i是当前子图的索引
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # 使用make_pipeline函数创建一个包含数据预处理步骤和分类器的管道
        # StandardScaler()是一个预处理步骤，用于标准化特征（即减去平均值并除以标准差）
        # 参数clf是分类器实例，用于分类
        clf = make_pipeline(StandardScaler(), clf)
        # 训练管道中的分类器
        # 使用训练集X_train和y_train来训练模型
        clf.fit(X_train, y_train)
        # 评估分类器在测试集X_test和y_test上的性能
        # score返回的是分类准确度
        score = clf.score(X_test, y_test)
        # 使用DecisionBoundaryDisplay.from_estimator方法绘制分类器的决策边界
        # 根据分类器clf和整个数据集X计算决策边界，在子图ax上绘制
        # cmap=cm设置颜色映射，alpha=0.8是透明度，eps=0.5是决策边界周围的额外边缘空间
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )
        # 在之前创建的子图上绘制训练集
        # X_train[:, 0] 和 X_train[:, 1] 是训练数据集的两个特征，用于确定每个数据点在图上的位置
        # c=y_train 指定每个点的颜色，基于训练集的标签确定的
        # cmap=cm_bright 是颜色映射，使用之前定义的红蓝颜色映射
        # edgecolors="k" 给每个点添加黑色边缘
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # 对测试数据集进行绘制
        # X_test[:, 0] 和 X_test[:, 1] 是测试数据集的两个特征
        # c=y_test 用于确定测试集数据点的颜色
        # alpha=0.6 设置数据点的透明度，可以区分训练集和测试集的数据点
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )
        # 设置当前子图ax的x轴和y轴的显示范围
        # x_min, x_max, y_min, y_max是之前计算的特征的最小和最大值，加上了一定的边界空间
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # 移除子图的x轴和y轴的刻度
        ax.set_xticks(())
        ax.set_yticks(())
        # 在每组子图的第一行设置标题，标题是分类器的名称，方便识别每列子图对应的分类器
        # ds_cnt是数据集的索引，当其为0时，表示当前处理的是第一个数据集
        if ds_cnt == 0:
            ax.set_title(name)
        # 在子图的右下角添加一个文本，内容是分类器在测试集上的得分（保留两位小数）
        # x_max - 0.3 和 y_min + 0.3 指定文本的位置，lstrip("0")移除分数前的零（“0.90”会显示为“.90”）
        # size=15 设置文本的大小，horizontalalignment="right"指定文本的对齐方式
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        # 将子图索引i增加1
        i += 1
# 调整子图的布局，使得图形中的元素不会重叠，整个图形更加美观
plt.tight_layout()
# 显示图形
plt.show()