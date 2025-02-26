#导入所需的库
import pandas as pd
import numpy as np
#导入数据集
train = pd.read_csv('D:/train.csv')
test = pd.read_csv('D:/test.csv')
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['缺失数', '缺失率'])
missing_data

train.Name = train.Name.fillna('Unknown')
print(train.isnull().sum())

train['CryoSleep']=train['CryoSleep'].fillna(False)
train['VIP']=train['VIP'].fillna(False)
print(train.isnull().sum())

train['Age']=train['Age'].fillna(train['Age'].mean())
print(train.isnull().sum())

train['RoomService']=train['RoomService'].fillna(train['RoomService'].mean())
train['FoodCourt']=train['FoodCourt'].fillna(train['FoodCourt'].mean())
train['ShoppingMall']=train['ShoppingMall'].fillna(train['ShoppingMall'].mean())
train['Spa']=train['Spa'].fillna(train['Spa'].mean())
train['VRDeck']=train['VRDeck'].fillna(train['VRDeck'].mean())
print(train.isnull().sum())

analys = train.loc[:,['HomePlanet','Destination']]
analys['numeric'] =1
analys.groupby(['Destination','HomePlanet']).count()

train['Destination']=train['Destination'].fillna('TRAPPIST-1e')
train['HomePlanet']=train['HomePlanet'].fillna('Earth')
print(train.isnull().sum())

#导入所需的库
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#画出热力图
# 选择只包含数值型数据的列
numeric_columns = train.select_dtypes(include=['float64', 'int64'])

# 计算相关系数矩阵
corrmat = numeric_columns.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True)
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(18, 12)
fig.subplots_adjust(wspace=0.3, hspace=0.3)

temp = train.fillna(-1)

sns.barplot(x="HomePlanet", y="Transported", data=temp, ax=ax[0][0])
sns.barplot(x="CryoSleep", y="Transported", data=temp, ax=ax[0][1])
sns.barplot(x="VIP", y="Transported", data=temp, ax=ax[1][0])
sns.barplot(x="Destination", y="Transported", data=temp, ax=ax[1][1])

#画出年龄分布图
plt.figure(figsize=(10, 5))
sns.histplot(data=train, x='Age', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)');

# 删除非数值型的列
train_numeric = train.select_dtypes(include=['float64', 'int64'])
# 记录 'Transported' 列
transported_column = train['Transported']
# 使用独热编码进行转换
train_encoded = pd.get_dummies(train_numeric)
# 将 'Transported' 列添加回 DataFrame
train_encoded['Transported'] = transported_column
corrmat = train_encoded.corr()
k = 6
high_corr_values = corrmat.nlargest(k, 'Transported')['Transported'].index
high_corr_values = high_corr_values.drop('Transported')

#查看high_corr_values
high_corr_values

Index(['FoodCourt', 'ShoppingMall', 'Age', 'VRDeck', 'Spa'], dtype='object')

#导入所需的库
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import metrics

#定义所需的参数
X = train[high_corr_values]
y = train['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#使用HistGradientBoostingClassifier模型进行预测并输出精度
from sklearn.ensemble import HistGradientBoostingClassifier
hgbc = HistGradientBoostingClassifier()
hgbc.fit(X_train,y_train)
hgbc_pred = hgbc.predict(X_test)
print("Hist gradient boosting accuracy: {}".format(metrics.accuracy_score(y_test,hgbc_pred)))

# 查看测试集
test

# 导出文件
test_ids = test["PassengerId"]

from sklearn.preprocessing import LabelEncoder

categorical_values_test = test.select_dtypes(include=['object']).columns

for i in categorical_values_test:
    lbl = LabelEncoder()
    lbl.fit(list(test[i].values))
    test[i] = lbl.transform(list(test[i].values))

# 由于HistGradientBoostingClassifier模型预测精度最高，因此使用HistGradientBoostingClassifier所预测的文件
real_predictions = hgbc.predict(test[high_corr_values])
print(len(test))
print(len(test.PassengerId))

test["PassengerId"] = test_ids

real_predictions = list(map(bool, real_predictions))

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': real_predictions})
output.to_csv('C:/Users/MyPC/Desktop/submission.csv', index=False)