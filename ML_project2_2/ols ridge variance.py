"""
=========================================================
Ordinary Least Squares and Ridge Regression Variance
=========================================================
Due to the few points in each dimension and the straight
line that linear regression uses to follow these points
as well as it can, noise on the observations will cause
great variance as shown in the first plot. Every line's slope
can vary quite a bit for each prediction due to the noise
induced in the observations.

Ridge regression is basically minimizing a penalised version
of the least-squared function. The penalising `shrinks` the
value of the regression coefficients.
Despite the few data points in each dimension, the slope
of the prediction is much more stable and the variance
in the line itself is greatly reduced, in comparison to that
of the standard linear regression

"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
# 用于普通最小二乘法和岭回归
from sklearn import linear_model
# 使用numpy的c_方法创建一个列向量，并用.T对其进行转置，生成一个训练数据的2x1矩阵:[[0.5],[1]]
X_train = np.c_[0.5, 1].T
# 对应的目标值向量，每个数据点都有一个
y_train = [0.5, 1]
# 创建了一个测试数据的2x1矩阵:[[0],[2]]，并用.T对其进行转置
X_test = np.c_[0, 2].T
# 设置随机数种子为0
np.random.seed(0)
# 一个字典，包含两个键值对
# 每个键代表一个分类器名称，与其相对应的值是该分类器的实例
# ols代表普通最小二乘法，linear_model.LinearRegression()创建一个普通最小二乘法线性回归模型的实例
# ridg代表岭回归，linear_model.Ridge(alpha=0.1)创建一个岭回归模型的实例。alpha是正则化的强度
classifiers = dict(
    ols=linear_model.LinearRegression(), ridge=linear_model.Ridge(alpha=0.1)
)
# 遍历之前定义的classifiers字典
# 在每次迭代中，name变量将是分类器的名称（即"ols"或"ridge"），clf是对应的模型实例
for name, clf in classifiers.items():
    # 使用matplotlib库创建一个新的绘图窗口
    # 窗口的大为4x3英寸
    # subplots()方法创建一个新的子图
    # fig 用于控制整个图像的属性，ax用于控制子图的属性
    fig, ax = plt.subplots(figsize=(4, 3))
    # 运行6次循环
    for _ in range(6):
        # 生成一个带有噪声的训练数据集
        # np.random.normal(size=(2, 1))生成一个2x1的随机矩阵，其中的元素来自标准正态分布
        # 这个随机矩阵乘以0.1（缩小噪声的幅度），然后加到原始的X_train数据上
        this_X = 0.1 * np.random.normal(size=(2, 1)) + X_train
        # 用上一步生成的带有噪声的训练数据this_X和目标值y_train训练模型
        clf.fit(this_X, y_train)
        # 使用模型对测试数据X_test进行预测，将预测结果绘制为灰色的线
        # X_test是x轴的坐标，clf.predict(X_test)是y轴的坐标
        ax.plot(X_test, clf.predict(X_test), color="gray")
        # 在图上以灰色的点表示带有噪声的训练数据this_X
        # .scatter()方法用于绘制散点图
        # y_train是y轴的坐标
        # s=3设置点的大小，marker="o"定义点的形状为圆形，zorder=10确保这些点始终在线的上方
        # zoeder参数指定绘图元素的顺序，值越大，绘制的顺序越靠后，也就是越靠近顶层
        ax.scatter(this_X, y_train, s=3, c="gray", marker="o", zorder=10)
    # 使用X_train训练数据和y_train目标值来训练模型
    clf.fit(X_train, y_train)
    # 预测X_test的测试数据，并将结果绘制成蓝色线，线宽为2
    ax.plot(X_test, clf.predict(X_test), linewidth=2, color="blue")
    # 使用红色的"+"标记在图上绘制训练数据点
    # 参数s=30设置标记的大小，zorder=10确保这些点始终在线的上方
    ax.scatter(X_train, y_train, s=30, c="red", marker="+", zorder=10)
    # 设置子图的标题为当前模型的名称
    ax.set_title(name)
    # 设置x轴的范围为0到2
    ax.set_xlim(0, 2)
    # 设置y轴的范围为0到1.6
    ax.set_ylim((0, 1.6))
    # 设置x轴的标签为"X"
    ax.set_xlabel("X")
    # 设置y轴的标签为"y"
    ax.set_ylabel("y")
    # 调整子图布局，使之不会重叠
    fig.tight_layout()
# 显示图形
plt.show()