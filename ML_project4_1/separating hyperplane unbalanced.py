import matplotlib.pyplot as plt
# 用于支持向量机算法
from sklearn import svm
# 用于生成数据集
from sklearn.datasets import make_blobs
# 用于绘制决策边界
from sklearn.inspection import DecisionBoundaryDisplay

# 生成数据集时每个簇的样本数
n_samples_1 = 1000
n_samples_2 = 100
# 列表包含两个元素，每个元素都是一个坐标点
# 代表两个簇的中心位置
centers = [[0.0, 0.0], [2.0, 2.0]]
# 列表包含两个数值
# 代表每个簇的标准差，用于控制簇内数据点的分散程度
clusters_std = [1.5, 0.5]
# 调用 make_blobs 函数来生成数据
# X 和 y，分别代表生成的特征数据和对应的标签
# n_samples=[n_samples_1, n_samples_2]: 指定每个簇的样本数量，即第一个簇1000个样本，第二个簇100个样本
# centers=centers: 指定簇的中心位置
# cluster_std=clusters_std: 指定每个簇的标准差
# random_state=0: 设置随机数种子
# shuffle=False: 指定生成的数据是否被随机打乱。False 表示数据将按照生成的顺序排列
X, y = make_blobs(
    n_samples=[n_samples_1, n_samples_2],
    centers=centers,
    cluster_std=clusters_std,
    random_state=0,
    shuffle=False,
)

# 创建一个 SVC 对象，命名为 clf
# SVC是支持向量机用于分类的版本
# kernel="linear": 指定使用线性核函数
# C=1.0: 设置正则化参数 C
clf = svm.SVC(kernel="linear", C=1.0)
# 训练 clf 模型
# X 是特征数据，y 是相应的标签
clf.fit(X, y)
# 创建另一个 SVC 对象，命名为 wclf，这次使用类权重
# class_weight={1: 10}: 给类别 1 赋予更高的权重（比例为10:1）。类别 1 的样本将被视为类别 0 样本的10倍
wclf = svm.SVC(kernel="linear", class_weight={1: 10})
# 训练 wclf 模型，使用和 clf.fit 相同的数据
wclf.fit(X, y)
# 使用matplotlib的 scatter 方法绘制数据点
# X[:, 0], X[:, 1]: 提取 X 的所有行但只有前两列
# c=y: 指定点的颜色，根据标签 y 来着色
# cmap=plt.cm.Paired: 指定颜色映射，使用的matplotlib的 Paired 色彩映射。Paired 色彩映射包含12种颜色，用于着色12个不同的类别
# edgecolors="k": 设置点的边缘颜色为黑色
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

# 获取当前活跃的matplotlib轴对象，并将其存储在变量 ax 中
# .gca() 方法用于获取当前活跃的matplotlib轴对象
ax = plt.gca()
# 使用 DecisionBoundaryDisplay.from_estimator 方法绘制由之前训练的SVM模型（clf）定义的决策边界
# clf: 之前创建并训练的SVM模型
# X: 特征数据，用于确定决策边界应该在哪里绘制
# plot_method="contour": 指定绘图方法为等高线（contour）
# colors="k": 设置等高线的颜色为黑色
# levels=[0]: 指定等高线的水平。 [0] 表示绘制函数值为0的等值线，代表SVM的决策边界
# alpha=0.5: 设置等高线的透明度为半透明
# linestyles=["-"]: 设置等高线的样式为实线
# ax=ax: 指定绘制等高线的轴对象
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[0],
    alpha=0.5,
    linestyles=["-"],
    ax=ax,
)

# 用于 wclf 模型
# wclf: 之前创建并训练的具有类权重的SVM模型
# X: 特征数据，同样用于确定决策边界的位置
# plot_method="contour": 指定绘图方法为等高线（contour）
# colors="r": 设置等高线的颜色为红色
# levels=[0]: 指定等高线的水平，同样是 [0]，表示绘制函数值为0的等值线，代表SVM的决策边界
# alpha=0.5: 设置等高线的透明度为半透明
# linestyles=["-"]: 设置等高线的样式为实线
# ax=ax: 使用之前获取的同一个轴对象 ax 进行绘制
wdisp = DecisionBoundaryDisplay.from_estimator(
    wclf,
    X,
    plot_method="contour",
    colors="r",
    levels=[0],
    alpha=0.5,
    linestyles=["-"],
    ax=ax,
)

# 添加一个图例
# [disp...[0], wdisp...[0]]: 从之前创建的决策边界显示对象 disp 和 wdisp 中提取绘制的等高线（决策边界）
# disp.surface_ 和 wdisp.surface_ 是 DecisionBoundaryDisplay 对象的属性，包含绘图的相关信息
# .collections[0] 提取绘制的第一个等高线集合，分别对应于 clf 和 wclf 模型的决策边界
# ["non weighted", "weighted"]: 字符串列表，提供两个决策边界的描述，用于图例中的标签
# "non weighted" 对应于没有类权重的SVM模型 clf，"weighted" 对应于有类权重的SVM模型 wclf
# loc="upper right": 指定图例的位置，放置在图表的右上角
plt.legend(
    [disp.surface_.collections[0], wdisp.surface_.collections[0]],
    ["non weighted", "weighted"],
    loc="upper right",
)
# 显示图表
plt.show()