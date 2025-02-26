import numpy as np
#导入numpy工具包
from os import listdir
from sklearn.neural_network import MLPClassifier

def img2vector(fileName):
    retMat = np.zeros([1024], int)
    fr = open(fileName)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i * 32 + j] = lines[i][j]
    return retMat

def readDataSet(path):
    fileList = listdir(path)
    numFiles = len(fileList)
    dataSet = np.zeros([numFiles, 1024], int)
    hwLabels = np.zeros([numFiles, 10])
    for i in range(numFiles):
        filePath=fileList[i]
        digit = int(filePath.split('_')[0])
        hwLabels[i][digit] = 1.0
        dataSet[i] = img2vector(path + '/' + filePath)
    return dataSet, hwLabels

train_dataSet,train_hwLabels =readDataSet('dataset3/dataset/trainingDigits')

clf= MLPClassifier(hidden_layer_sizes=(100,),
                   activation='logistic',solver='adam',
                   learning_rate_init= 0.0001,max_iter=2000)

clf.fit(train_dataSet,train_hwLabels)

dataSet,hwLabels =readDataSet('dataset3/dataset/testDigits')

res =clf.predict(dataSet)
error_num=0
num=len(dataSet)
for i in range(num):
    if np.sum(res[i] == hwLabels[i]) < 10:
        error_num+=1
print("Total num:",num,"Wrong num:",\
      error_num,"WrongRate:",error_num/float(num))