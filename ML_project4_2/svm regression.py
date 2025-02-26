import matplotlib.pyplot as plt
import numpy as np
# S支持向量机用于回归的版本
from sklearn.svm import SVR

# 生成一个随机数数组作为特征数据 X
# np.random.rand(40, 1): 生成一个40行1列的数组，数组中的元素是[0, 1)区间内的随机数
# 5 *: 将数组中的每个元素乘以5，使得数据范围变为[0, 5)
# np.sort(..., axis=0): 对数组按照列（axis=0）进行排序
X = np.sort(5 * np.random.rand(40, 1), axis=0)
# 生成目标变量 y
# np.sin(X): 计算 X 中每个元素的正弦值
# .ravel(): 将结果展平为一维数组
y = np.sin(X).ravel()
# 为目标变量 y 添加噪声
# y[::5]: 选择 y 中每第5个元素
# np.random.rand(8): 生成8个[0, 1)区间的随机数
# 0.5 - np.random.rand(8): 生成中心在0.5的随机噪声。0.5- 是为了使得噪声的范围变为[-0.5, 0.5)
# 3 *: 放大噪声的幅度
# +=: 将噪声加到选定的 y 元素上
y[::5] += 3 * (0.5 - np.random.rand(8))

# 创建一个使用径向基函数（RBF）核的SVR模型
# kernel="rbf": 使用径向基函数核
# C=100: 设置正则化参数为100
# gamma=0.1: 设置核函数的参数
# epsilon=0.1: 设置epsilon参数，用于指定SVR模型中的epsilon边距
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
# 创建一个使用线性核的SVR模型
# kernel="linear": 使用线性核
# gamma="auto": 自动选择gamma参数
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
# 创建一个使用多项式核的SVR模型
# kernel="poly": 使用多项式核
# degree=3: 多项式核的度数为3
# coef0=1: 多项式核的独立项系数
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

# 绘图时线的宽度
lw = 2

# 创建一个包含三个SVR模型的列表 svrs
svrs = [svr_rbf, svr_lin, svr_poly]
# 创建一个列表，包含三个字符串，用于标识上述三个SVR模型所使用的核类型
kernel_label = ["RBF", "Linear", "Polynomial"]
# 创建一个列表，包含三个颜色代码
# 用于绘制每个SVR模型的输出
model_color = ["m", "c", "g"]

# 使用matplotlib的 subplots 函数创建一个图表和三个子图轴（axes）
# nrows=1, ncols=3: 指定子图布局为1行3列
# figsize=(15, 10): 设置整个图表的大小为宽度15英寸，高度10英寸
# sharey=True: 表明所有子图共享y轴的刻度和范围
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
# 使用 enumerate 函数遍历 svrs 列表，同时提供元素的索引 (ix) 和元素值 (svr)
# svrs 列表包含三个SVR模型
for ix, svr in enumerate(svrs):
    # 对于每个SVR模型，使用 plot 方法在对应的子图上绘制预测曲线
    # X: 特征数据，用于SVR模型的预测
    # svr.fit(X, y).predict(X): 使用 .fit(X, y) 方法训练SVR模型，然后使用 .predict(X) 方法对相同的特征数据 X 进行预测。生成预测结果，用于绘图
    # color=model_color[ix]: 设置曲线的颜色，使用之前定义的 model_color 列表中对应的颜色
    # lw=lw: 设置曲线的宽度，使用之前定义的 lw 变量
    # label=...: 设置曲线的标签，使用格式化字符串来插入对应的核函数标签（例如 "RBF model", "Linear model"）
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    # 在每个SVR模型对应的子图上绘制散点图，显示该模型的支持向量
    # X[svr.support_]: 选择特征数据 X 中对应于支持向量的元素。svr.support_ 属性包含支持向量在数据集中的索引
    # y[svr.support_]: 选择目标变量 y 中对应于支持向量的元素
    # facecolor="none": 设置散点的填充颜色为无，使得散点呈现为空心
    # edgecolor=model_color[ix]: 设置散点的边缘颜色，使用之前定义的 model_color 列表中对应的颜色
    # s=50: 设置散点的大小
    # label=...: 设置散点图的标签，使用格式化字符串来插入对应的核函数标签，并添加 "support vectors" 字样，例如 "RBF support vectors"
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=50,
        label="{} support vectors".format(kernel_label[ix]),
    )
    # 在每个SVR模型对应的子图上绘制散点图，显示非支持向量的训练数据点
    # X[...]: 选择 X 中不是支持向量的数据点
    # np.arange(len(X)) 生成一个与 X 长度相同的整数序列，np.setdiff1d 计算两个数组的差集，用于找出不在 svr.support_ 中的索引
    # y[...]: 选择 y 中对应于非支持向量的数据点
    # facecolor="none": 设置散点的填充颜色为无，使得散点呈现为空心
    # edgecolor="k": 设置散点的边缘颜色为黑色
    # s=50: 设置散点的大小
    # label="...": 设置散点图的标签
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="other training data",
    )
    # 为每个子图添加图例
    # loc="upper center": 设置图例的位置为子图上方居中
    # bbox_to_anchor=(0.5, 1.1): 用于指定图例的锚点。(0.5, 1.1) 表示图例的中心位于子图上边缘之外的位置
    # ncol=1: 设置图例中的列数为1，即图例条目垂直排列
    # fancybox=True: 设置图例边框为圆角
    # shadow=True: 给图例添加阴影效果
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

# 在图表的底部中心位置添加水平轴的标签
# 0.5, 0.04: 指定文本的位置，将文本置于图表底部中心（水平方向居中，垂直方向接近底部）
# "data": 文本内容，即轴标签
# ha="center", va="center": 设置水平和垂直对齐方式为居中
fig.text(0.5, 0.04, "data", ha="center", va="center")
# 在图表的左侧中心位置添加垂直轴的标签
# 0.06, 0.5: 指定文本的位置，将文本置于图表左侧中心（垂直方向居中，水平方向接近左边）
# "target": 文本内容，即轴标签
# rotation="vertical": 设置文本旋转为垂直。将文本旋转90度，文本内容从左向右显示
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
# 设置整个图表的标题
# fontsize=14: 设置标题的字体大小为14
fig.suptitle("Support Vector Regression", fontsize=14)
# 显示整个图表
plt.show()