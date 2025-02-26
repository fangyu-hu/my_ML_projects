# 用于加载鸢尾花数据集
from sklearn.datasets import load_iris
# 从 sklearn 库加载鸢尾花数据集
iris = load_iris()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
# 用于绘制决策边界
from sklearn.inspection import DecisionBoundaryDisplay
# 决策树分类器
from sklearn.tree import DecisionTreeClassifier


# 定义类别的数量
n_classes = 3
# 用于绘图的颜色代码
# "r", "y", "b" 分别代表红色、黄色、蓝色
plot_colors = "ryb"
# 设置绘图的步长，用于在绘制决策边界时定义网格的密度
plot_step = 0.02
# 循环，遍历所有可能的特征对组合
# 鸢尾花数据集有四个特征，这里创建所有可能的两两特征组合（共 6 组）
# pairidx 是当前迭代的特征对的索引，pair 是当前迭代的特征对
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # 从鸢尾花数据集中选取当前迭代的特征对
    # iris.data[:, pair] 表示从数据集中选取所有样本（:）的特定两个特征（pair）
    X = iris.data[:, pair]
    # 获取数据集的目标标签
    y = iris.target
    # 初始化一个决策树分类器，使用选取的特征对 X 和目标标签 y 来训练
    clf = DecisionTreeClassifier().fit(X, y)
    # 创建子图
    # 创建一个 2 行 3 列的子图网格，并定位到当前迭代的位置
    # pairidx + 1 是当前迭代的位置
    ax = plt.subplot(2, 3, pairidx + 1)
    # 调整子图的布局，确保子图之间的间距适当
    # h_pad=0.5, w_pad=0.5, pad=2.5 分别设置子图之间的水平间距、垂直间距、整个图形的边距
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    # DecisionBoundaryDisplay.from_estimator 用于根据训练好的估计器（clf，即决策树分类器）和给定的数据（X，即特定的特征对）绘制决策边界
    # clf 是之前训练好的决策树模型
    # X 是当前迭代中选择的特征对的数据
    # cmap=plt.cm.RdYlBu 设置用于绘制决策边界的颜色映射，从红色（Red）到黄色（Yellow）再到蓝色（Blue）的渐变
    # response_method="predict" 指定用于生成决策边界的方法，使用模型的 predict 方法
    # ax=ax 指定了要在哪个子图上绘制决策边界
    # xlabel 和 ylabel 分别设置子图的 x 轴和 y 轴的标签，使用的是鸢尾花数据集中对应特征的名称
    # iris.feature_names 是鸢尾花数据集中所有特征的名称。pair[0] 和 pair[1] 分别是当前迭代的特征对的两个特征的索引
    # iris.feature_names[pair[0]] 和 iris.feature_names[pair[1]] 分别是当前迭代的特征对的两个特征的名称
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )

    # 循环，遍历每个类别及其对应的颜色
    # n_classes 是类别数（3），plot_colors 是之前定义的颜色代码（"ryb"，代表红色、黄色、蓝色）
    # range() 函数用于生成一个整数序列，从 0 到 n_classes - 1
    # zip() 函数用于将两个序列（range() 函数生成的整数序列和 plot_colors）中的元素一一对应
    # i 是当前迭代的类别的索引，color 是当前迭代的类别对应的颜色
    for i, color in zip(range(n_classes), plot_colors):
        # 使用 np.where 找出目标数组 y（标签数组）中等于当前类别 i 的索引
        # 这些索引用于从特征数组 X 中选取对应类别的数据点
        idx = np.where(y == i)
        # plt.scatter 用于绘制散点图
        # X[idx, 0] 和 X[idx, 1] 分别是选定类别的数据点在两个特征轴上的坐标
        # X[idx, 0] 表示选定类别的数据点在第一个特征轴上的坐标
        # c=color 指定散点的颜色
        # label=iris.target_names[i] 设置图例标签为当前类别的名称
        # iris.target_names 是鸢尾花数据集中所有类别的名称
        # cmap=plt.cm.RdYlBu 再次设置颜色映射
        # edgecolor="black" 设置数据点的边缘颜色为黑色
        # s=15 设置散点的大小
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

# 设置图形的标题
plt.suptitle("Decision surface of decision trees trained on pairs of features")
# 添加图例
# loc="lower right" 指定图例放置在图形的右下角
# borderpad 和 handletextpad 设置图例的边界和文本的间距，都设置为 0
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
# 调整坐标轴的范围，紧密地围绕数据
_ = plt.axis("tight")

# 用于绘制决策树结构图
from sklearn.tree import plot_tree
# 创建一个新的图形
plt.figure()
# 初始化一个新的决策树分类器，使用整个鸢尾花数据集（所有特征和目标标签）来训练
# iris.data 是鸢尾花数据集中的所有特征
# iris.target 是鸢尾花数据集中的所有目标标签
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
# 绘制决策树的结构图
# filled=True 指示用不同的颜色填充决策树的节点，不同的颜色代表不同的类别
plot_tree(clf, filled=True)
# 设置标题
plt.title("Decision tree trained on all the iris features")
# 显示图形
plt.show()