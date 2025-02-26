# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image

#First we read and flatten the image.
#导入图片，将数据转化为RGB的百分比
original_img = np.array(Image.open('dataset4/dataset/tree.jpg'),dtype=np.float64)/255
#查看原始图片三维(800,600,3)
print("图片维度形状：",original_img.shape)
plt.imshow(original_img)
plt.axis('off')
#获取图片数据的维度/形状(长、宽、像素)，图片扁平化（三维转换为二维）
original_dimensions = tuple(original_img.shape)
width,height,depth = tuple(original_img.shape)
image_flattened = np.reshape(original_img,(width * height,depth))
print("图片扁平化后形状",image_flattened.shape)

#打乱图片像素，随机选取18阳阳个颜色样本
image_array_sample = shuffle(image_flattened,random_state=0)[:1000]
#聚类为64类，每个聚类都将成为压缩调色板中的一种颜色
estimator = KMeans(n_clusters=64,random_state=0)
estimator.fit(image_array_sample)

#为原图中每个像素预测其应该分配到哪个聚类中
cluster_assignments = estimator.predict(image_flattened)

#通过压缩调色盘和聚类分配结果创建压缩图片
compressed_palette=estimator.cluster_centers_#获取聚类中心，每行为RGB的值，64个聚类点
print("聚类中心：",compressed_palette)
compressed_img = np.zeros((width,height,compressed_palette.shape[1]))
label_idx=0
for i in range(width):
    for j in range(height):
        compressed_img[i][j] = compressed_palette[cluster_assignments[label_idx]]
        label_idx += 1
plt.figure()#创建新图形窗口
plt.subplot(121)
plt.title('Original Image',fontsize=24)
plt.imshow(original_img)
plt.axis('off')
plt.subplot(122)
plt.title('Compressed Image',fontsize=24)
plt.imshow(compressed_img)
plt.axis('off')
plt.show()