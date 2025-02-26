import matplotlib.pyplot as plt
import numpy as np
# 决策树回归器
from sklearn.tree import DecisionTreeRegressor

# 创建一个 numpy 的随机数生成器实例，设置种子为 1
# 用于生成随机数
rng = np.random.RandomState(1)
# 生成一个 80x1 的随机数数组，范围在 0 到 5 之间（rng.rand(80, 1) 生成 0 到 1 之间的随机数，乘以 5 后变为 0 到 5）
# 使用 np.sort(..., axis=0) 对数组按列进行排序
X = np.sort(5 * rng.rand(80, 1), axis=0)
# 通过 np.sin(X) 生成对应于 X 的正弦值，作为目标变量 y
# ravel() 将数组变为一维数组，用于创建目标向量
y = np.sin(X).ravel()
# 对目标变量 y 的每第五个元素进行修改，以引入一些噪声

# y[::5]：步长切片操作，选取 y 数组中每隔五个元素的值，即第 0、5、10、15... 个元素
# rng.rand(16): 生成一个包含 16 个从 0 到 1 之间的随机数的数组
# 0.5 - rng.rand(16)：将生成的随机数从每个数中减去 0.5。生成的新随机数数组将在 -0.5 到 +0.5 之间
# 3 * (0.5 - rng.rand(16))：将生成的数组中的每个元素乘以 3。扩大了随机数的范围，使其位于 -1.5 到 +1.5 之间。增加了噪声的幅度
# y[::5] += 3 * (0.5 - rng.rand(16))
#   将生成的噪声值加到 y 数组的特定元素上（即每隔五个元素）
#   在目标变量 y 中引入周期性的噪声
y[::5] += 3 * (0.5 - rng.rand(16))

# 初始化一个决策树回归器 regr_1，设置最大深度为 2
regr_1 = DecisionTreeRegressor(max_depth=2)
# 初始化另一个决策树回归器 regr_2，设置最大深度为 5
regr_2 = DecisionTreeRegressor(max_depth=5)
# 使用之前准备的训练数据（X 和 y）分别训练这两个回归器
regr_1.fit(X, y)
regr_2.fit(X, y)

# 创建一个测试数据集 X_test，范围从 0.0 到 5.0，步长为 0.01
# np.arange 生成一个一维数组，[:, np.newaxis] 将其转换为二维数组

# np.arange(0.0, 5.0, 0.01): 创建一个从 0.0 开始到 5.0 结束的一维数组，步长为 0.01
# [:, np.newaxis]：将 np.arange 生成的一维数组转换成二维数组
# : 表示选取一维数组中的所有元素。np.newaxis 是 NumPy 中的一个特殊索引，用于增加一个新的轴维度
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# 使用两个训练好的回归器 regr_1 和 regr_2 分别对测试数据 X_test 进行预测，将预测结果存储在 y_1 和 y_2 中
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# 创建一个新的 图形
plt.figure()
# plt.scatter 绘制散点图，用于显示原始数据点
# X 和 y 是原始的特征和目标数据
# s=20 设置散点的大小
# edgecolor="black" 设置散点的边缘颜色为黑色
# c="darkorange" 设置散点的颜色为深橙色
# label="data" 为这些散点设置图例标签
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot 绘制一条线，表示 max_depth=2 的决策树回归器的预测结果
# X_test 是测试数据，y_1 是对应的预测结果
# color="cornflowerblue" 设置线的颜色为某种蓝色
# label="max_depth=2" 为这条线设置图例标签
# linewidth=2 设置线宽
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# 绘制 max_depth=5 的决策树回归器的预测结果
# color="yellowgreen" 设置线的颜色为黄绿色
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# 设置 x 轴和 y 轴的标签
plt.xlabel("data")
plt.ylabel("target")
# 设置图形的标题
plt.title("Decision Tree Regression")
# 添加图例
plt.legend()
# 显示图形
plt.show()