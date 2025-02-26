# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save &amp; Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv("C:/Users/administrator1/Desktop/train_titanic.csv")
train_data.head()

test_data = pd.read_csv("C:/Users/administrator1/Desktop/test_titanic.csv")
test_data.head()

b=0
for i in range(8693):
    b=b+train_data.iloc[i]['Age']
b=b/8693
print(b)

a=0
for i in range(8693):
    if train_data.iloc[i]['VIP']==True and train_data.iloc[i]['Transported']==True:
        a=a+1
print(a/8692)

a=0
for i in range(8693):
    if train_data.iloc[i]['CryoSleep']==True and train_data.iloc[i]['Transported']==True:
        a=a+1
print(a/8692)

a=0
for i in range(8693):
    if train_data.iloc[i]['HomePlant']=='Earth' and train_data.iloc[i]['Transported']==True:
        a=a+1
print(a/8692)

from sklearn.ensemble import RandomForestClassifier

y = train_data["Transported"]

features = ["HomePlanet", "CryoSleep", "VIP"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predictions})
output.to_csv('submission1.csv', index=False)
print("Your submission was successfully saved!")