#%load_ext autotime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
plt.style.use('ggplot')

# 读取训练集数据
train_data = pd.read_csv('C:/Users/administrator1/Desktop/train.csv')

# 读取测试集数据
test_data = pd.read_csv('C:/Users/administrator1/Desktop/test.csv')

plt.figure(figsize=(12,8))
sns.heatmap(train_data .corr(), annot=True, fmt='.2f', cmap='PuBu')

train_data.corr()['price_range'].sort_values()

sns.pairplot(train_data[["px_width", "px_height", "battery_power", "ram", "price_range"]], hue="price_range")

X_train, y_train = train_data[train_data.columns.delete(-1)], train_data['price_range']
X_test, y_test = test_data[test_data.columns.delete(-1)], test_data['price_range']

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
coef = linear_model.coef_#回归系数
line_pre = linear_model.predict(X_test)
print('SCORE:{:.4f}'.format(linear_model.score(X_test, y_test)))
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, line_pre))))
coef

df_coef = pd.DataFrame()
df_coef['Title'] = train_data.columns.delete(-1)
df_coef['Coef'] = coef
df_coef

line_pre = linear_model.predict(X_test)  #预测值
hos_pre = pd.DataFrame()
hos_pre['Predict'] = line_pre
hos_pre['Truth'] = y_test
hos_pre_sampled = hos_pre[::10]  # 每隔10个样本绘制一个点
hos_pre_sampled.plot(figsize=(18, 8))

plt.scatter(y_test, line_pre,label='y_test')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4,label='predicted')

#%load_ext autotime

from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X_train, y_train)
tree_reg_pre = tree_reg.predict(X_test)#预测值
print('SCORE:{:.4f}'.format( tree_reg.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,tree_reg_pre))))#RMSE(标准误差)

tree_reg_pre= tree_reg.predict(X_test)  # 决策树模型的预测值
hos_pre = pd.DataFrame()
hos_pre['Predict'] = tree_reg_pre
hos_pre['Truth'] = y_test
hos_pre_sampled = hos_pre[::10]  # 每隔10个样本绘制一个点
hos_pre_sampled.plot(figsize=(18, 8))

plt.scatter(y_test,tree_reg_pre,label='y')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4,label='predicted')







from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import r2_score as r2, mean_squared_error as mse, mean_absolute_error as mae


def svr_model(kernel):
    svr = SVR(kernel=kernel)
    svr.fit(X_train, y_train)
    y_predict = svr.predict(X_test)

    # score(): Returns the coefficient of determination R^2 of the prediction.
    print(kernel, ' SVR的默认衡量评估值值为：', svr.score(X_test, y_test))
    print(kernel, ' SVR的R-squared值为：', r2(y_test, y_predict))
    print(kernel, ' SVR的均方误差（mean squared error）为：', mse(y_test, y_predict))
    print(kernel, ' SVR的平均绝对误差（mean absolute error）为：', mae(y_test, y_predict))
    # print(kernel,' SVR的均方误差（mean squared error）为：',mse(scalery.inverse_transform(y_test), scalery.inverse_transform(y_predict)))
    # print(kernel,' SVR的平均绝对误差（mean absolute error）为：',mae(scalery.inverse_transform(y_test),scalery.inverse_transform(y_predict)))

    return svr

linear_svr = svr_model(kernel='linear')

# 使用 svr_model 函数获取训练好的 SVM 模型
svm_model = svr_model(kernel='linear')

# 创建 DataFrame 用于绘图
svm_pre_df = pd.DataFrame()
svm_pre_df['Predict'] = svm_model.predict(X_test)
svm_pre_df['Truth'] = y_test

# 抽样
svm_pre_sampled = svm_pre_df[::10]

# 绘制图形
plt.figure(figsize=(18, 8))
plt.plot(svm_pre_sampled['Predict'], label='Predicted')
plt.plot(svm_pre_sampled['Truth'], label='True')
plt.title('SVM Prediction vs True Values')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend()
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 选择需要的特征
selected_features = ['px_height', 'px_width', 'ram', 'battery_power']
X2 = np.array(train_data[selected_features])
y = train_data['price_range']

# 划分训练集和测试集
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, random_state=1, test_size=0.2)

# 创建线性回归模型
linear_model2 = LinearRegression()

# 训练模型
linear_model2.fit(X2_train, y_train)

# 输出模型的截距和系数
print('Intercept:', linear_model2.intercept_)
print('Coefficients:', linear_model2.coef_)

# 预测值
line2_pre = linear_model2.predict(X2_test)

# 模型评价
print('SCORE:{:.4f}'.format(linear_model2.score(X2_test, y_test)))
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, line2_pre))))