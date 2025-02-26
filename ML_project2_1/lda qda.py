"""
====================================================================
Linear and Quadratic Discriminant Analysis with covariance ellipsoid
====================================================================

This example plots the covariance ellipsoids of each class and
decision boundary learned by LDA and QDA. The ellipsoids display
the double standard deviation for each class. With LDA, the
standard deviation is the same for all the classes, while each
class has its own standard deviation with QDA.

"""

# %%
# Colormap
# --------

# 将 matplotlib 起名为 mpl
import matplotlib as mpl
import matplotlib.pyplot as plt
# 用于定义颜色
from matplotlib import colors
# 定义颜色映射，用于将数据值映射到颜色
# colors.LinearSegmentedColormap用于创建线性分段的颜色映射
cmap = colors.LinearSegmentedColormap(
    # 颜色映射的名称
    "red_blue_classes",
    # 字典，键是颜色通道（红、绿、蓝），值是关于该通道如何变化的列表
    {
        # 定义了红色通道的变化
        # 列表中的每个元组代表一个锚点，第一个值是数据值（从0到1），第二个和第三个值是颜色的开始和结束范围
        # 在数据值为0时，红色的范围从1开始并保持为1，在数据值为1时，红色的范围从0.7开始并结束于0.7
        "red": [(0, 1, 1), (1, 0.7, 0.7)],
        # 定义绿色通道的变化
        "green": [(0, 0.7, 0.7), (1, 0.7, 0.7)],
        # 定义蓝色通道的变化
        "blue": [(0, 0.7, 0.7), (1, 1, 1)],
    },
)
# 使用matplotlib的register_cmap方法注册定义的颜色映射，用来在后续的绘图中使用
# cmap=cmap指定了要注册的颜色映射
plt.cm.register_cmap(cmap=cmap)
import numpy as np
# 用于生成两个具有相同协方差矩阵的高斯分布样本
def dataset_fixed_cov():
    """Generate 2 Gaussians samples with the same covariance matrix"""
    # n代表每个高斯分布样本的大小，300
    # dim代表数据的维度，2维
    n, dim = 300, 2
    # 设置numpy的随机数生成器的种子为0
    np.random.seed(0)
    # 定义一个2x2的数组C，代表协方差矩阵
    # np.array 用于创建数组
    C = np.array([[0.0, -0.23], [0.83, 0.23]])
    # 定义一个变量X，包含生成的高斯样本
    # np.r_用于按行堆叠数组
    X = np.r_[
        # 使用np.random.randn(n, dim)生成一个大小为n x dim的随机样本，其元素服从标准正态分布（均值为0，标准差为1）
        # np.random.randn 用于生成服从标准正态分布的样本
        # 使用np.dot函数与协方差矩阵C进行点乘，产生具有特定协方差结构的样本
        # np.dot 用于计算两个数组的点积
        np.dot(np.random.randn(n, dim), C),
        # 与上一行类似，但在生成的样本上加了一个偏移[1, 1]
        # 第二个高斯分布的中心移动到了[1, 1]
        # + np.array([1, 1]) 用于将两个数组相加
        np.dot(np.random.randn(n, dim), C) + np.array([1, 1]),
    ]
    # 定义一个标签数组y
    # np.zeros(n)生成一个包含n个0的数组，np.ones(n)生成一个包含n个1的数组
    # np.hstack将它们堆叠在一起，形成一个长度为2*n的数组
    # np.hstack用于按列堆叠数组
    y = np.hstack((np.zeros(n), np.ones(n)))
    # 返回生成的样本X和其对应的标签y
    return X, y
# 用于生成两个具有不同协方差矩阵的高斯分布样本
def dataset_cov():
    """Generate 2 Gaussians samples with different covariance matrices"""
    # n代表每个高斯分布样本的大小，300
    # dim代表数据的维度，2维
    n, dim = 300, 2
    # 设置numpy的随机数生成器的种子为0
    np.random.seed(0)
    # 定义一个2x2的数组C，基于给定的2x2矩阵乘以2.0
    # 代表协方差矩阵
    C = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
    # 包含生成的高斯样本
    # np.r_用于按行堆叠数组
    X = np.r_[
        # 使用np.random.randn(n, dim)生成一个大小为n x dim的随机样本，其元素服从标准正态分布（均值为0，标准差为1）
        # np.random.randn 用于生成服从标准正态分布的样本
        # 使用np.dot函数与协方差矩阵C进行点乘，产生具有特定协方差结构的样本
        # np.dot 用于计算两个数组的点积
        np.dot(np.random.randn(n, dim), C),
        # 与前一行相似，但这里使用C.T，即矩阵C的转置，作为协方差矩阵
        # 使这一组样本与上一组协方差结构不同
        # 加上np.array([1, 4])为这组样本加了一个偏移
        np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4]),
    ]
    # 定义一个标签数组y
    # np.zeros(n)生成一个包含n个0的数组，np.ones(n)生成一个包含n个1的数组
    # np.hstack将它们堆叠在一起，形成一个长度为2*n的数组
    # np.hstack用于按列堆叠数组
    y = np.hstack((np.zeros(n), np.ones(n)))
    # 返回生成的样本X和其对应的标签y
    return X, y

# 用于线性代数运算
from scipy import linalg
# 根据给定的参数绘制数据
# lda用于分类的模型
# X数据点，y真实标签，y_pred预测的标签，fig_index子图的索引
def plot_data(lda, X, y, y_pred, fig_index):
    # 使用matplotlib的subplot函数在2x2的网格中创建一个子图
    # fig_index决定子图在网格中的位置
    splot = plt.subplot(2, 2, fig_index)
    # 如果fig_index等于1
    if fig_index == 1:
        # 为子图设置标题
        plt.title("Linear Discriminant Analysis")
        # 为子图设置y轴的标签
        # 使用\n换行，使标签在两行上显示
        plt.ylabel("Data with\n fixed covariance")
    # 如果fig_index等于2
    elif fig_index == 2:
        # 为子图设置标题
        plt.title("Quadratic Discriminant Analysis")
    # 如果fig_index等于3
    elif fig_index == 3:
        # 为子图设置y轴的标签
        plt.ylabel("Data with\n varying covariances")
    # 创建一个布尔数组tp（True Positive），每个元素都表示真实标签y是否等于预测标签y_pred
    # 如果某个元素的真实标签与预测标签匹配，则该位置为True，否则为False
    tp = y == y_pred  # True Positive
    # 为两个类别（标签0和标签1）分别创建了True Positive的布尔数组
    # tp0包含了标签为0的样本的True Positive结果
    # tp1包含了标签为1的样本的True Positive结果

    # tp[y == 0]：使用一个条件索引。首先，y == 0会为y数组中的每一个元素返回一个布尔值：如果元素值为0，则返回True；否则返回False。产生一个与y数组大小相同的布尔数组
    # 这个布尔数组被用来从tp数组中筛选出对应位置为True的元素。只有那些与布尔数组中True值位置相对应的tp中的元素会被选择
    # tp[y == 1]：同理，为y数组中的每一个元素返回一个布尔值：如果元素值为1，则返回True；否则返回False。然后，这个布尔数组用来从tp数组中筛选出对应位置为True的元素
    # tp0, tp1 = ... 最后，将标签为0的True Positive结果赋值给tp0，将标签为1的True Positive结果赋值给tp1
    tp0, tp1 = tp[y == 0], tp[y == 1]
    # 根据真实标签y的值将数据X分为两组：
    # X0包含标签为0的样本，X1包含标签为1的样本
    X0, X1 = X[y == 0], X[y == 1]
    # 将X0（标签为0的样本）分为两组：X0_tp包含正确分类的样本，X0_fp包含被错误分类的样本
    # ~tp0是tp0的逻辑否定，表示False Positive
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    # 将X1（标签为1的样本）分为正确分类和错误分类的两组
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    # class 0: dots
    # 使用matplotlib的scatter函数绘制X0_tp中的数据点（正确分类为0的样本）
    # 这些点的x坐标是X0_tp[:, 0]，y坐标是X0_tp[:, 1]
    # markder表示点的标记样式为.，color表示颜色为红色
    # [:, 0]表示取所有行的第0列，[:, 1]表示取所有行的第1列
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color="red")
    # 使用scatter函数绘制X0_fp中的数据点（被错误分类为1的样本）
    # markder点的标记样式为x，s表示点的大小为20，颜色为深红色（#990000）
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker="x", s=20, color="#990000")  # dark red
    # class 1: dots
    # 使用matplotlib的scatter函数绘制X1_tp中的数据点（正确分类为1的样本）
    # 这些点的x坐标是X1_tp[:, 0]，y坐标是X1_tp[:, 1]
    # 点的标记样式为.，颜色为蓝色
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker=".", color="blue")
    # 使用scatter函数绘制X1_fp中的数据点（被错误分类为0的样本）
    # 这些点的标记样式为x，大小为20，颜色为深蓝色（#000099）
    plt.scatter(
        X1_fp[:, 0], X1_fp[:, 1], marker="x", s=20, color="#000099"
    )  # dark blue
    # class 0 and 1 : areas
    # 分别代表在x轴和y轴上的采样点的数量
    nx, ny = 200, 100
    # 获取当前图形的x轴范围，并将最小值赋给x_min，最大值赋给x_max
    # plt.xlim()用于获取当前图形的x轴范围返回一个元组，元组的第一个元素是最小值，第二个元素是最大值。
    x_min, x_max = plt.xlim()
    # 获取当前图形的y轴范围，并将最小值赋给y_min，最大值赋给y_max
    y_min, y_max = plt.ylim()
    # 使用np.meshgrid和np.linspace函数创建一个网格
    # 这个网格覆盖了当前图形的x和y范围，并在这两个方向上均匀地采样
    # np.linspace(x_min, x_max, nx)返回一个长度为nx的数组，其中包含了从x_min到x_max的均匀间隔的数字
    # np.meshgrid函数将这两个数组转换为两个二维数组，分别包含了x轴和y轴上的采样点
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    # 使用分类器lda的predict_proba方法预测网格上每个点的类别概率

    # xx.ravel() 和 yy.ravel():
        # 这两个调用使用ravel()方法来将二维数组（网格的x和y坐标）平坦化为一维数组
    # np.c_[xx.ravel(), yy.ravel()]:
        # 使用numpy的特殊索引语法进行数组的列式组合
        # np.c_ 会将两个一维数组按列组合，形成一个二维数组，其中第一列是xx.ravel()的值，第二列是yy.ravel()的值
    # lda.predict_proba(...):
        # 使用分类器（lda）的predict_proba方法
        # 返回每个样本属于每个类别的概率
        # 输入是组合后的网格坐标，为每一个网格点预测其属于每个类别的概率
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # 从预测的概率中选择第二列（即类别为1的概率），并将其形状重新塑造为与xx相同
    # [:, 1]表示取所有行的第1列
    # .reshape(xx.shape)将数组的形状重新塑造为与xx相同
    Z = Z[:, 1].reshape(xx.shape)
    # 使用pcolormesh方法绘制概率区域图
    # 显示一个颜色渐变的区域，颜色表示一个点被分类为类别1的概率
    # 使用定义的red_blue_classes颜色映射和规范化方法

    # plt.pcolormesh(...):
    # 用于在二维平面上绘制一个伪彩色图。用来显示一个二维数组的值，每个值都被映射到一个颜色上
    # xx, yy:
    # 二维网格上的X和Y坐标。定义了想要绘制的区域的形状和范围
    # Z:
    # 一个二维数组，其中的值表示在对应的xx和yy坐标上的颜色强度或值。Z包含每个网格点被分类为类别1的概率
    # cmap="red_blue_classes":
    # 定义了一个颜色映射，决定了如何将Z中的值转换为颜色
    # 使用之前定义的red_blue_classes的颜色映射，表示低概率是红色，高概率是蓝色，中间值则是过渡颜色
    # norm=colors.Normalize(0.0, 1.0):
    # 定义了一个颜色规范化对象，决定如何将Z中的值映射到颜色映射上
    # 0.0到1.0表示Z中的值被规范化为这个范围内，并且这个范围的值被映射到cmap上
    # zorder=0:
    # 决定绘图元素的绘制顺序。数值较小的元素会被绘制在底层，数值较大的元素会被绘制在顶层
    # zorder设置为0，表示这个伪彩色图会被绘制在其他图形元素的底部
    plt.pcolormesh(
        xx, yy, Z, cmap="red_blue_classes", norm=colors.Normalize(0.0, 1.0), zorder=0
    )
    # 使用contour方法绘制概率为0.5的轮廓线
    # 这条线表示模型认为两个类别的概率相等的地方

    # plt.contour(...):用于绘制轮廓线
    # xx, yy, Z:
    # 要绘制轮廓的数据。xx和yy代表网格上每个点的x和y坐标，Z代表与这些坐标对应的数据值
    # [0.5]:要绘制轮廓线的数据值的列表
    # 在代码中只有一个值，即0.5，意味着只会绘制Z值为0.5的地方的轮廓线
    # linewidths=2.0:轮廓线的宽度
    # colors="white":廓线的颜色是白色
    plt.contour(xx, yy, Z, [0.5], linewidths=2.0, colors="white")
    # means
    # 在图上绘制一个点
    # 这个点的坐标是lda.means_[0]，即lda计算的类别0的均值坐标
    # lda.means_[0][0], lda.means_[0][1]分别是均值的x和y坐标
    # 点的标记样式为*，颜色为黄色，大小为15，边缘颜色为灰色
    plt.plot(
        lda.means_[0][0],
        lda.means_[0][1],
        "*",
        color="yellow",
        markersize=15,
        markeredgecolor="grey",
    )
    # 绘制类别1的均值坐标
    # 其样式与上面的点相同
    plt.plot(
        lda.means_[1][0],
        lda.means_[1][1],
        "*",
        color="yellow",
        markersize=15,
        markeredgecolor="grey",
    )
    return splot
# 在给定的子图splot上绘制一个椭圆
# splot：子图
# mean：椭圆的中心位置
# cov：协方差矩阵
# color：椭圆的填充颜色
def plot_ellipse(splot, mean, cov, color):
    # 使用linalg.eigh函数计算协方差矩阵cov的特征值和特征向量
    # v是特征值数组，w是对应的特征向量矩阵
    v, w = linalg.eigh(cov)
    # 取特征向量矩阵w的第一个向量，并将其除以其范数，得到一个单位向量u

    # w[0]:从w中取出的第一个特征向量。w[0]表示取w的第一行，即第一个特征向量
    # linalg.norm(w[0]): 使用linalg.norm函数计算w[0]的范数。linalg是numpy的模块，提供线性代数的功能。norm函数计算向量的欧几里得范数
    # u = w[0] / linalg.norm(w[0]):
    # 将第一个特征向量w[0]除以其范数。结果是一个单位向量u，其方向与原始特征向量w[0]相同，但其长度为1
    u = w[0] / linalg.norm(w[0])
    # 计算单位向量u的角度
    # 这个角度描述了椭圆的主轴与x轴的夹角
    # u[1] 和 u[0]: 从单位向量u中取出的两个元素。u[0]代表向量在x轴上的分量，u[1]代表向量在y轴上的分量
    # u[1] / u[0]:计算向量u在y轴分量与x轴分量的比值。表示向量u与x轴之间的切线斜率
    # np.arctan(...):是numpy库中的反正切函数。接受一个参数返回介于-π/2到π/2之间的角度值。在代码中，返回的是向量u与x轴之间的夹角
    angle = np.arctan(u[1] / u[0])
    # 将角度从弧度转换为度

    # np.pi:numpy库中提供的π的近似值
    # angle / np.pi:将原先以弧度为单位的angle值除以π，得到一个新的值，表示angle相对于π的比例
    # 180 * angle / np.pi: 将上一步得到的比例转换为度数。一个完整的圆（2π弧度）对应360度，1π弧度对应180度。乘以180后就可以得到angle对应的度数值
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    # 使用mpl.patches.Ellipse创建一个椭圆形状
    # mean：椭圆的中心位置
    # 2 * v[0] ** 0.5：椭圆的主轴长度，是第一个特征值v[0]的平方根的2倍。表示2个标准差的距离
    # 2 * v[1] ** 0.5：椭圆的次轴长度，与主轴的计算方法相同
    # angle=180 + angle：椭圆的旋转角度。加180度是为了调整椭圆的方向
    # facecolor=color：椭圆的填充颜色，由函数参数传入
    # edgecolor="black"：椭圆的边界颜色设为黑色
    # linewidth=2：设置椭圆的边界线宽为2
    ell = mpl.patches.Ellipse(
        mean,
        2 * v[0] ** 0.5,
        2 * v[1] ** 0.5,
        angle=180 + angle,
        facecolor=color,
        edgecolor="black",
        linewidth=2,
    )
    # 设置椭圆的裁剪框为splot（即子图）的边界框
    # 确保椭圆不会超出子图的范围
    ell.set_clip_box(splot.bbox)
    # 设置椭圆的透明度为0.2，使得椭圆半透明，便于观察椭圆背后的其他图形
    ell.set_alpha(0.2)
    # 将椭圆ell添加到子图splot上，使其显示在图上
    # add_artist方法用于将一个matplotlib的图形元素添加到图形中
    splot.add_artist(ell)
    # 将子图splot的x轴刻度设置为空，x轴上不会显示任何刻度
    splot.set_xticks(())
    # 将子图splot的y轴刻度设置为空
    splot.set_yticks(())
# 用于在给定的子图（splot）上根据线性判别分析（LDA）的结果绘制协方差椭圆
# lda是LDA的实例，splot是子图的实例
def plot_lda_cov(lda, splot):
    # 使用plot_ellipse函数为LDA的第一个类别绘制红色的协方差椭圆
    # lda.means_[0]是LDA计算的第一个类别的均值
    # lda.covariance_是LDA计算的协方差矩阵
    # "red"是椭圆的填充颜色
    plot_ellipse(splot, lda.means_[0], lda.covariance_, "red")
    # 为LDA的第二个类别绘制蓝色的协方差椭圆
    # lda.means_[1]是LDA计算的第二个类别的均值
    # lda.covariance_是LDA计算的协方差矩阵
    # "blue"是椭圆的填充颜色
    plot_ellipse(splot, lda.means_[1], lda.covariance_, "blue")
# 在给定的子图上根据二次判别分析（QDA）的结果绘制协方差椭圆
# qda是QDA的实例，splot是子图的实例
def plot_qda_cov(qda, splot):
    # 使用plot_ellipse函数为QDA的第一个类别绘制红色的协方差椭圆
    # qda.means_[0]是QDA计算的第一个类别的均值
    # qda.covariance_[0]是QDA计算的第一个类别的协方差矩阵
    # "red"是椭圆的填充颜色
    plot_ellipse(splot, qda.means_[0], qda.covariance_[0], "red")
    # 为QDA的第二个类别绘制蓝色的协方差椭圆
    # qda.means_[1]是QDA计算的第二个类别的均值
    # qda.covariance_[1]是QDA计算的第二个类别的协方差矩阵
    plot_ellipse(splot, qda.means_[1], qda.covariance_[1], "blue")
# 创建一个新的图形窗口
# figsize指定图形的尺寸为宽10英寸、高8英寸，facecolor指定图形背景颜色为白色
plt.figure(figsize=(10, 8), facecolor="white")
# 为整个图形设置一个标题
# y=0.98表示标题的垂直位置接近图的顶部，fontsize=15指定字体大小为15
# y表示标题的垂直位置，0表示图的底部，1表示图的顶部
plt.suptitle(
    "Linear Discriminant Analysis vs Quadratic Discriminant Analysis",
    y=0.98,
    fontsize=15,
)
# 线性判别分析（LDA）和二次判别分析（QDA）
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
# for循环迭代两个数据集：一个是dataset_fixed_cov()生成的，具有固定协方差的数据
# 另一个是dataset_cov()生成的，具有不同协方差的数据
# enumerate函数返回当前循环的索引（i）和值（在此处是数据集的特征X和标签y）
for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
    # Linear Discriminant Analysis
    # 使用LDA模型创建一个实例
    # solver="svd"表示使用奇异值分解方法进行计算
    # store_covariance=True表示存储计算的协方差矩阵
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    # 使用LDA的fit方法训练模型，然后用predict方法预测训练集的分类结果
    # lda.fit(X, y)用于训练模型，X是训练集的特征，y是训练集的标签
    # lda.predict(X)用于预测训练集的标签，X是训练集的特征
    # 预测的标签保存在y_pred中
    y_pred = lda.fit(X, y).predict(X)
    # 调用plot_data函数，在子图上绘制数据并返回该子图的实例
    # lda是LDA的实例，X是数据点，y是真实标签，y_pred是预测标签，fig_index=2 * i表示子图的索引
    # fig_index确保每次迭代都在新的子图上绘制
    splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
    # 使用plot_lda_cov函数为LDA的每个类别绘制协方差椭圆
    plot_lda_cov(lda, splot)
    # 使用axis函数调整坐标轴范围，使其适应数据的大小，使图像看起来更整洁
    # tight表示紧密适应数据的大小
    plt.axis("tight")
    # Quadratic Discriminant Analysis
    # 创建一个二次判别分析（QDA）的实例
    # store_covariance=True来存储计算的协方差矩阵
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    # 使用QDA的fit方法训练模型，并使用predict方法对训练集进行预测
    # 预测的结果被保存在y_pred中
    y_pred = qda.fit(X, y).predict(X)
    # 再次调用plot_data函数，在子图上显示QDA的结果
    # qda是QDA的实例，X是数据点，y是真实标签，y_pred是预测标签，fig_index=2 * i + 2表示子图的索引
    # fig_index确保每次迭代都在新的子图上绘制
    splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
    # 使用plot_qda_cov函数为QDA的每个类别绘制协方差椭圆
    plot_qda_cov(qda, splot)
    # 调整坐标轴的范围，使其紧密适应数据的大小
    plt.axis("tight")
# 自动调整子图参数，使得子图之间有足够的空间，并且与图的边缘也有足够的空间
plt.tight_layout()
# 调整子图布局，top=0.92确保标题与最上面的子图之间有足够的空间
# top表示子图布局的顶部位置
plt.subplots_adjust(top=0.92)
# 显示整个图像
plt.show()
