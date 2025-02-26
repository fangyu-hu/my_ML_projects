import matplotlib.pyplot as plt
import numpy as np
# 用于支持向量机算法
from sklearn import svm

# 用于可视化分类器在二维空间中的决策边界
# classifier: 用于做出决策的分类器
# sample_weight: 样本的权重
# axis: 用于绘图的matplotlib轴对象
# title: 绘制图表的标题
def plot_decision_function(classifier, sample_weight, axis, title):
    # 生成一个网格，用于绘制决策边界
    # np.linspace(-4, 5, 500): 创建一个从-4到5的500个等间隔的点的序列。定义网格在x轴和y轴上的范围和密度
    # np.linspace 用于创建一个等间隔的数值序列
    # np.meshgrid(...): 根据这两个一维数组生成两个二维网格，xx 和 yy 分别代表网格的x坐标和y坐标
    # np.meshgrid 用于生成网格点坐标矩阵
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

    # 计算分类器的决策函数值
    # np.c_[...]: 将 xx 和 yy 的网格数据平铺并组合成一个二维数组，每行代表网格上一个点的x和y坐标
    # .ravel(): 将二维数组转换为一维数组
    # np.c_ 用于将两个一维数组组合成一个二维数组
    # classifier.decision_function(...): 调用分类器的 decision_function 方法，计算每个网格点的决策函数值。用于判断数据点属于哪个类别
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # 将决策函数的计算结果 Z 重塑回与 xx 相同的形状
    Z = Z.reshape(xx.shape)

    # 使用 contourf 方法绘制决策边界
    # xx, yy: 网格的X和Y坐标
    # Z: 决策函数的值，用于确定决策边界
    # alpha=0.75: 设置填充颜色的透明度
    # cmap=plt.cm.bone: 选择颜色映射为 bone，这是一种灰度色图
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    # 在同一轴上绘制散点图，显示训练数据点
    # X[:, 0], X[:, 1]: 从数据集 X 中提取X和Y坐标
    # c=y: 以目标变量 y 作为散点的颜色标签
    # s=100 * sample_weight: 设置散点的大小，基于 sample_weight 来加权
    # 这使得具有更大权重的样本在图上显示更大
    # alpha=0.9: 设置散点的透明度
    # cmap=plt.cm.bone: 设置和决策边界相同的颜色映射
    # edgecolors="black": 设置散点的边缘颜色为黑色
    axis.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=100 * sample_weight,
        alpha=0.9,
        cmap=plt.cm.bone,
        edgecolors="black",
    )

    # 关闭当前轴的坐标轴，图表中不显示任何坐标轴标记
    axis.axis("off")
    # 设置当前轴的标题，使用函数传入的 title 参数
    axis.set_title(title)

# 设置随机数生成器的种子为0
np.random.seed(0)
# 生成特征数据集 X
# np.random.randn(10, 2) + [1, 1]: 生成10个二维的正态分布随机数，然后每个数都加上 [1, 1]。结果是一个随机点集，集中在点 (1, 1) 附近
# np.random.randn(10, 2): 又生成另外10个二维的正态分布随机数，这些点集中在原点 (0, 0) 附近
# np.r_[...]: 将这两个点集合并成一个20个点的数据集
X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
# 创建目标变量 y
# 前10个样本标记为1，后10个样本标记为-1，对应于之前创建的两组数据点
# [1] * 10: 创建一个长度为10的数组，每个元素都是1
# [-1] * 10: 创建一个长度为10的数组，每个元素都是-1
# +: 将两个数组合并成一个数组
y = [1] * 10 + [-1] * 10
# 生成一个随机的样本权重数组
# 使用 np.random.randn(len(X)) 生成一个与 X 相同长度的随机数数组，然后取其绝对值，确保权重是正的
sample_weight_last_ten = abs(np.random.randn(len(X)))
# 创建一个恒定的样本权重数组
# 每个样本的权重都设为1
# np.ones 用于创建一个全为1的数组
sample_weight_constant = np.ones(len(X))
# 将最后5个样本的权重增加到原来的5倍。给特定样本更多的重视
# [15:]：取数组的后15个元素
# *= 5: 将后15个元素乘以5
sample_weight_last_ten[15:] *= 5
# 将第10个样本的权重增加到原来的15倍。给予特定样本特别高的权重
sample_weight_last_ten[9] *= 15

# 创建一个SVM分类器对象 clf_no_weights
# svm.SVC(gamma=1): 使用 SVC 类构建一个支持向量机分类器。gamma 参数设置为1
clf_no_weights = svm.SVC(gamma=1)
# 使用特征数据 X 和标签 y 训练分类器 clf_no_weights
clf_no_weights.fit(X, y)
# 创建另一个SVM分类器对象 clf_weights，其设置与 clf_no_weights 相同
clf_weights = svm.SVC(gamma=1)
# 使用相同的特征数据 X 和标签 y 训练分类器 clf_weights，但这次传入样本权重 sample_weight_last_ten
clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)

# 使用matplotlib的 subplots 函数创建一个图表和两个子图轴（axes）
# 1, 2: 指定子图布局为1行2列
# figsize=(14, 6): 设置整个图表的大小为宽度14英寸，高度6英寸
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 用 plot_decision_function 函数绘制 clf_no_weights（不考虑样本权重的分类器）的决策边界
# clf_no_weights: 不考虑样本权重的分类器
# sample_weight_constant: 对应于 clf_no_weights 的样本权重，这里是恒定权重
# axes[0]: 指定在第一个子图上绘制
# "Constant weights": 设置子图的标题
plot_decision_function(
    clf_no_weights, sample_weight_constant, axes[0], "Constant weights"
)
# 使用 plot_decision_function 函数，但这次是绘制 clf_weights（考虑样本权重的分类器）的决策边界
# clf_weights: 考虑样本权重的分类器
# sample_weight_last_ten: 对应于 clf_weights 的样本权重，这里是修改过的权重
# axes[1]: 指定在第二个子图上绘制
# "Modified weights": 设置子图的标题
plot_decision_function(clf_weights, sample_weight_last_ten, axes[1], "Modified weights")

# 显示图形
plt.show()