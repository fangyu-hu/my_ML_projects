
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer #新版本0.20以来 SimpleImputer 代替 Imputer,故代码有所变动
# 导入自动生成训练集和测试集的模块train_test_split
from sklearn.model_selection import train_test_split

# 导入预测结果评估模块classification_report
from sklearn.metrics import classification_report

# 依次导入3个分类器模块
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# 数据导入函数
def load_dataset(feature_paths, label_paths): # 传入两个参数：特征文件的列表和标签文件的列表
    feature = np.ndarray(shape=(0,41)) #列数量和特征维度一致为41;
    label = np.ndarray(shape=(0,1)) # 列数量与标签维度一致为1

    for file in feature_paths:
        df = pd.read_table(file,delimiter=',',na_values='?',header=None)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean') #将数据补全
        imp.fit(df) #训练预处理器
        df = imp.transform(df) #生成预处理结果
        feature = np.concatenate((feature,df)) # 将新读入的数据合并到特征集中，依次遍历完所有特征文件

    for file in label_paths:
        df = pd.read_table(file,header=None)
        label = np.concatenate((label,df))
    label = np.ravel(label)
    return feature,label


if __name__ == '__main__':
    featurePaths = ['dataset1/dataset1/A/A.feature', 'dataset1/dataset1/B/B.feature', 'dataset1/dataset1/C/C.feature', 'dataset1/dataset1/D/D.feature', 'dataset1/dataset1/E/E.feature']
    labelPaths = ['dataset1/dataset1/A/A.feature', 'dataset1/dataset1/B/B.feature', 'dataset1/dataset1/C/C.feature', 'dataset1/dataset1/D/D.feature', 'dataset1/dataset1/E/E.feature']
    x_train,y_train = load_dataset(featurePaths[:4],labelPaths[:4])
    x_test,y_test = load_dataset(featurePaths[4:], labelPaths[4:])
    x_train,x_,y_train,y_ = train_test_split(x_train,y_train,train_size=0.2)

    # 主函数——knn
    from sklearn.neighbors import KNeighborsClassifier, BallTree

    print("Start training knn")

    # 使用BallTree算法，替代KNeighborsClassifier中的默认算法
    knn = KNeighborsClassifier(algorithm='ball_tree')
    knn.fit(x_train, y_train)

    print("Training done!")

    answer_knn = knn.predict(x_test)
    print("Prediction done!")

    # 主函数——决策树
    print("Start training DT")
    dt = DecisionTreeClassifier().fit(x_train,
                                      y_train)
    print("Training done!")
    answer_dt = dt.predict(x_test)
    print("Prediction done!")

    # 主函数——贝叶斯
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')

    # 主函数——分类结果分析
    # 使用classification_report函数对分类结果，从精确率precision、召回率recall、f1值f1-scor和支持度support四个维度进行衡量。
    # 输出
    print("\n\nThe classification report for knn:")
    print(classification_report(y_test, answer_knn))

    print("\n\nThe classification report for dt:")
    print(classification_report(y_test, answer_dt))

    print("\n\nThe classification report for gnb:")
    print(classification_report(y_test, answer_gnb))