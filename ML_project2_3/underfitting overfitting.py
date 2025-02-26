"""
============================
Underfitting vs. Overfitting
============================

This example demonstrates the problems of underfitting and overfitting and
how we can use linear regression with polynomial features to approximate
nonlinear functions. The plot shows the function that we want to approximate,
which is a part of the cosine function. In addition, the samples from the
real function and the approximations of different models are displayed. The
models have polynomial features of different degrees. We can see that a
linear function (polynomial with degree 1) is not sufficient to fit the
training samples. This is called **underfitting**. A polynomial of degree 4
approximates the true function almost perfectly. However, for higher degrees
the model will **overfit** the training data, i.e. it learns the noise of the
training data.
We evaluate quantitatively **overfitting** / **underfitting** by using
cross-validation. We calculate the mean squared error (MSE) on the validation
set, the higher, the less likely the model generalizes correctly from the
training data.

"""

import matplotlib.pyplot as plt
import numpy as np
# 用于执行线性回归
from sklearn.linear_model import LinearRegression
# 用来执行交叉验证
from sklearn.model_selection import cross_val_score
# 将多个处理步骤组合成一个单一的对象
from sklearn.pipeline import Pipeline
# 用于生成多项式特征
from sklearn.preprocessing import PolynomialFeatures
# 为给定的输入X生成一个真实的函数值
def true_fun(X):
    # 对于给定的输入X，返回对应的余弦值
    # np.cos是NumPy库中的余弦函数，np.pi是π的值
    return np.cos(1.5 * np.pi * X)
# 设置了随机数生成器的种子为0
np.random.seed(0)
# 生成30个样本数据
n_samples = 30
# 将要使用的多项式的次数
degrees = [1, 4, 15]
# 生成了一个包含n_samples个随机数的数组，并对其进行排序
# np.random.rand(n_samples)生成一个在[0,1)区间的随机数数组
# np.sort将数组进行排序
X = np.sort(np.random.rand(n_samples))
# 生成了一个由真实函数true_fun计算的y值，并给它添加了一些噪音
# np.random.randn(n_samples)生成一个平均值为0、标准差为1的正态分布随机数数组，乘以0.1是降低噪音的幅度
y = true_fun(X) + np.random.randn(n_samples) * 0.1
# 使用matplotlib的plt.figure函数初始化一个新的图形窗口，并设置其大小为14x5英寸
plt.figure(figsize=(14, 5))
# 循环degrees列表中的元素数量次
for i in range(len(degrees)):
    # 创建一个子图
    # 图形有1行和len(degrees)列，当前绘制的是第i + 1个图
    ax = plt.subplot(1, len(degrees), i + 1)
    # 将当前子图的x轴和y轴的刻度设置为空，子图上不会显示任何刻度标签
    plt.setp(ax, xticks=(), yticks=())
    # 创建一个多项式特征生成器
    # 根据指定的度数degrees[i]生成多项式特征
    # 参数include_bias=False表示不包括偏置项
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    # 创建一个线性回归模型的实例
    linear_regression = LinearRegression()
    # 开始定义一个管道（Pipeline）。用来组合多个步骤，可以一次性执行
    pipeline = Pipeline(
        [
            # 第一个步骤是使用polynomial_features生成多项式特征
            ("polynomial_features", polynomial_features),
            # 第二个步骤，使用线性回归模型linear_regression来拟合通过多项式转换后的数据
            ("linear_regression", linear_regression),
        ]
    )
    # 使用管道来拟合数据
    # X[:, np.newaxis]将X转换成一个二维数组。np.newaxis在指定的轴上为X添加了一个新的维度
    # np.newaxis的作用就是在这一位置上给原数组增加一个维度，其中的值为1
    # 首先对数据X应用多项式特征转换，然后使用线性回归模型拟合转换后的数据和y
    pipeline.fit(X[:, np.newaxis], y)
    # Evaluate the models using crossvalidation
    # 使用cross_val_score方法来评估之前定义的pipeline模型
    scores = cross_val_score(
        # pipeline：使用之前定义的pipeline模型
        # X[:, np.newaxis]是输入数据，将X转换为一个二维数组
        # y是目标值
        # scoring：使用负均方误差(neg_mean_squared_error)作为评分标准
        # cv=10 使用10折交叉验证
        pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10
    )
    # 创建一个从0到1均匀分布的100个点的数组
    # np.linspace(0, 1, 100)生成一个在[0,1]区间的等差数列，包含100个元素
    X_test = np.linspace(0, 1, 100)
    # 使用模型对X_test进行预测，并在图上绘制预测的曲线，标签为"Model"
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    # 绘制真实函数的图形，标签为"True function"
    plt.plot(X_test, true_fun(X_test), label="True function")
    # 在图上绘制原始的数据点
    # edgecolor="b"表示点的边缘颜色是蓝色。大小是20，标签为"Samples"
    plt.scatter(X, y, edgecolor="b", s=20, label="Samples")
    # 设置x轴和y轴的标签为"x"和"y"
    plt.xlabel("x")
    plt.ylabel("y")
    # 设置x轴的范围从0到1，y轴的范围从-2到2
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    # 在图上添加图例，并将其位置设置为最佳位置
    # best：自动选择最佳位置
    plt.legend(loc="best")
    # 设置图的标题
    # 标题显示当前多项式的度数，以及模型的均方误差（MSE）的平均值和标准差
    # "Degree {}\nMSE = {:.2e}(+/- {:.2e})":这是一个带有占位符的字符串格式模板:
    # {} 用于插入变量值。\n 是一个换行符
    # {:.2e} 是一个格式说明符，表示数字应以指数（科学）格式显示，其中小数点后保留两位数字
    # .format(degrees[i], -scores.mean(), scores.std()):使用format()方法将具体的变量值插入到字符串模板中的占位符位置
    # degrees[i] 表示多项式的度数
    # -scores.mean() 计算模型的均方误差（MSE）的平均值
    # scores.std() 计算MSE的标准差
    plt.title(
        "Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()
        )
    )
# 显示绘制的图形
plt.show()