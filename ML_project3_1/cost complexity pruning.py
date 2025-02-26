import matplotlib.pyplot as plt
# 用于加载乳腺癌数据集
from sklearn.datasets import load_breast_cancer
# 用于将数据集分为训练集和测试集
from sklearn.model_selection import train_test_split
# 决策树分类器
from sklearn.tree import DecisionTreeClassifier


# load_breast_cancer 函数从 sklearn 库加载乳腺癌数据集
# return_X_y=True 参数返回特征矩阵 X 和目标向量 y
# X 包含数据集的特征，y 包含相应的标签（分类结果）
X, y = load_breast_cancer(return_X_y=True)
# train_test_split 函数用于将数据集分为训练集和测试集
# X_train 和 y_train 是训练集的特征和标签
# X_test 和 y_test 是测试集的特征和标签
# random_state=0 确保每次运行代码时数据集的分割方式相同
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 初始化一个决策树分类器 DecisionTreeClassifier
# random_state=0 用于确保每次运行模型时结果的可复现性
clf = DecisionTreeClassifier(random_state=0)
# cost_complexity_pruning_path 方法计算决策树在不同复杂度（α）下的成本复杂度剪枝路径
# 使用训练数据 X_train 和 y_train
# path 是一个包含两个关键数组的字典：ccp_alphas（α值）和 impurities（不纯度）
path = clf.cost_complexity_pruning_path(X_train, y_train)
# 提取 path 字典中的 ccp_alphas 和 impurities
# ccp_alphas 包含了不同剪枝级别的α值
# impurities 表示对应于这些α值的树的不纯度
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# 创建一个图表窗口
# fig 是整个图表对象，ax 是一个或多个子图中的一个，这里只创建了一个子图
fig, ax = plt.subplots()
# 绘制线图
# ccp_alphas[:-1] 和 impurities[:-1] 分别是 x 轴和 y 轴的数据
# [:-1] 表示取列表的除了最后一个元素之外的所有元素
# marker='o' 表示每个数据点都用圆圈标记
# drawstyle='steps-post' 指定绘图的风格，在每个数据点之后绘制阶梯
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
# 下面两行设置图表的 x 轴和 y 轴的标签
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
# 设置图表的标题
ax.set_title("Total Impurity vs effective alpha for training set")

# 用于存储训练后的决策树模型
clfs = []
# 遍历之前计算得到的 ccp_alphas 列表中的每个α值
for ccp_alpha in ccp_alphas:
    # 为每个α值创建一个新的 DecisionTreeClassifier 实例
    # ccp_alpha=ccp_alpha 设置决策树的剪枝参数
    # random_state=0 用于确保每次运行模型时结果的可复现性
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    # 使用训练集 X_train 和 y_train 训练决策树
    # fit 方法用于训练模型的
    clf.fit(X_train, y_train)
    # 将训练后的决策树 clf 添加到列表 clfs 中
    clfs.append(clf)
# 打印信息，展示最后一个决策树的节点数量和对应的 ccp_alpha 值
# clfs[-1] 是 clfs 列表中的最后一个元素，即最后一个训练的决策树
# clfs[-1].tree_.node_count 获取该树的节点数
# ccp_alphas[-1] 是 ccp_alphas 列表中的最后一个元素，即最大的α值
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

# 从 clfs（决策树列表）和 ccp_alphas（α值列表）中移除最后一个元素
# [:-1] 表示选择列表中除了最后一个元素之外的所有元素
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
# 使用列表推导式创建一个新列表 node_counts，包含每棵树的节点数
# clf.tree_.node_count 获取每棵决策树 clf 的节点数
node_counts = [clf.tree_.node_count for clf in clfs]
# 使用列表推导式创建一个新列表 depth，包含每棵树的最大深度
# clf.tree_.max_depth 获取每棵决策树的最大深度
depth = [clf.tree_.max_depth for clf in clfs]
# 创建一个图形和两个子图
# 2, 1 参数表示图形包含两行一列的子图
# ax 是一个包含两个子图的数组
fig, ax = plt.subplots(2, 1)
# 在第一个子图（ax[0]）上绘制了一个线图，展示 ccp_alphas（α值）与 node_counts（节点数）的关系
# marker="o" 表示用圆圈标记数据点，drawstyle="steps-post" 指定绘图的风格，在每个数据点之后绘制阶梯
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
# 下面三行设置第一个子图的 x 轴标签、y 轴标签、标题
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
# 在第二个子图（ax[1]）上，绘制 ccp_alphas（α值）与 depth（树深度）的关系
# marker="o" 表示用圆圈标记数据点，drawstyle="steps-post" 指定绘图的风格，在每个数据点之后绘制阶梯
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
# 下面三行设置第二个子图的 x 轴标签、y 轴标签、标题
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
# 调整子图的布局，确保图表元素不会重叠或压缩
fig.tight_layout()

# 使用列表推导式计算每个决策树 clf 在训练集 X_train 和 y_train 上的准确度，将结果存储在 train_scores 列表中
# clf.score() 方法返回模型的准确度，即正确分类的样本数占总样本数的比例
train_scores = [clf.score(X_train, y_train) for clf in clfs]
# 计算每个决策树在测试集 X_test 和 y_test 上的准确度，将结果存储在 test_scores 列表中
test_scores = [clf.score(X_test, y_test) for clf in clfs]
# 创建一个图形和一个子图
fig, ax = plt.subplots()
# 设置子图的 x 轴标签、y 轴标签、标题
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
# 分别绘制训练集和测试集的准确度与 alpha 值的关系
# marker="o" 表示用圆圈标记数据点，label 表示图例中的标签
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
# 添加图例，区分训练集和测试集的线条
ax.legend()
# 显示图表
plt.show()