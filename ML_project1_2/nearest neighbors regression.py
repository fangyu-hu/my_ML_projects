"""
============================
Nearest Neighbors regression
============================

Demonstrate the resolution of a regression problem
using a k-Nearest Neighbor and the interpolation of the
target using both barycenter and constant weights.

"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#
# License: BSD 3 clause (C) INRIA


# %%
# Generate sample data
# 生成样本数据
# --------------------
# 导入用到的所有库和模块
import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors

#设置随机数种子
np.random.seed(0)
# 生成样本数据
# 用 numpy 随机生成 40 个数据点，并对这40个点进行排序
X = np.sort(5 * np.random.rand(40, 1), axis=0)
# 后续预测用到一个从 0 到 5 的等间隔序列
T = np.linspace(0, 5, 500)[:, np.newaxis]
# 用sin(x)得到目标变量
# 响应
y = np.sin(X).ravel()

# Add noise to targets
# 对targets添加噪声，每隔5个点添加，噪声随机
y[::5] += 1 * (0.5 - np.random.rand(8))

# %%
# Fit regression model
# 适配回归模型
# --------------------
# 设置k-NN 邻近数量为5
n_neighbors = 5

# 用uniform和distance两种不同策略分别进行回归
for i, weights in enumerate(["uniform", "distance"]):
    # 模型初始化
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    # 模型拟合
    # 预测
    y_ = knn.fit(X, y).predict(T)

    # 用subplot函数绘制子图
    plt.subplot(2, 1, i + 1)
    # 绘制数据点
    plt.scatter(X, y, color="darkorange", label="data")
    # 绘制预测曲线
    plt.plot(T, y_, color="navy", label="prediction")
    # 绘制边界
    plt.axis("tight")
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

# 显示最终图像
plt.tight_layout()
plt.show()
