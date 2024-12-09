import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from PIL import Image
import os

def data_gen(N, d): #生成数据，N个数，d维度
    if d==2:    #2维
        mean = [-5, 5]
        cov = [[1, 0], [0, 0.01]]
    elif d==3:  #3维
        mean = [0, -6, 6]
        cov = [[1, 0, 0], [0, 0.01, 0], [0, 0, 1]]
    data = np.transpose(np.random.multivariate_normal(mean, cov, N))
    return data


def PCA(data, m):   #主成分分析
    #print(data.shape)
    N = data.shape[1]
    d = data.shape[0]
    mean = np.mean(data, axis=1) #compute row mean  #按行求均值
    data_centralized = np.zeros(data.shape) #d*N
    for i in range(d):
        data_centralized[i] = data[i] - mean[i] #减均值，去中心化
    #print(data_centralized)
    cov = np.dot(data_centralized, np.transpose(data_centralized))  #计算协方差矩阵
    #print(cov.shape)
    eigenValues, eigenVectors = np.linalg.eig(cov)  #特征向量，特征值
    #print(eigenValues)
    #print(eigenVectors)
    sortedIndex = np.argsort(eigenValues)   #特征值排序
    #print(sortedIndex)
    optimalVec = np.zeros((d, m), dtype='complex_')
    for j in range(m):  #选择m个特征值最大的特征向量
        optimalVec[:, j] = eigenVectors[:, (sortedIndex[d-j-1])]
    optimalVec = np.real(optimalVec)
    #print(optimalVec)
    project = np.dot(np.transpose(optimalVec), data_centralized)  #m*d d*N  #向轴上投影
    pca_data = np.zeros(data.shape, dtype='complex_')
    for i in range(d):  #重构数据
        pca_data[i] = np.dot(optimalVec[i], project) + mean[i]
    #print(pca_data)
    #pca_data = (optimalVec.dot(((data.T - mean).dot(optimalVec)).T)).T+ mean
    #print(pca_data)
    #print(face_pca)
    return pca_data, optimalVec, data_centralized, mean

def PCA_figure(d, data, pca_data, mean, optimalVec):    #画图
    if d == 2:
        fig, ax = plt.subplots()
        ax.scatter(data[0], data[1], facecolor="#00CED1", alpha = 0.6, label="Origin Data")
        ax.scatter(pca_data[0], pca_data[1], facecolor='#DC143C', alpha = 0.6, label='PCA Data')
        x = [mean[0] - 3 * optimalVec[0], mean[0] + 3 * optimalVec[0]]
        y = [mean[1] - 3 * optimalVec[1], mean[1] + 3 * optimalVec[1]]
        ax.plot(x, y, color='blue', label='eigenVector direction', alpha=0.5)
        ax.set_title('origin_data And PCA_data', fontsize=16)
        ax.set_xlabel('$x$', fontdict={'size': 14, 'color': 'black'})
        ax.set_ylabel('$y$', fontdict={'size': 14, 'color': 'black'})
    elif d == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data[0], data[1], data[2], facecolor="#00CED1", alpha = 0.6, label='Origin Data')
        ax.scatter(pca_data[0], pca_data[1], pca_data[2], facecolor='#DC143C', alpha = 0.6, label='PCA Data')
        # 画出2条eigen Vector 方向直线
        x = [mean[0] - 3 * optimalVec[0, 0], mean[0] + 3 * optimalVec[0, 0]]
        y = [mean[1] - 3 * optimalVec[1, 0], mean[1] + 3 * optimalVec[1, 0]]
        z = [mean[2] - 3 * optimalVec[2, 0], mean[2] + 3 * optimalVec[2, 0]]
        ax.plot(x, y, z, color='blue', label='eigenVector1 direction', alpha=1)
        x2 = [mean[0] - 3 * optimalVec[0, 1], mean[0] + 3 * optimalVec[0, 1]]
        y2 = [mean[1] - 3 * optimalVec[1, 1], mean[1] + 3 * optimalVec[1, 1]]
        z2 = [mean[2] - 3 * optimalVec[2, 1], mean[2] + 3 * optimalVec[2, 1]]
        ax.plot(x2, y2, z2, color='purple', label='eigenVector2 direction', alpha=1)

        ax.set_title('origin_data And PCA_data', fontsize=16)
        ax.set_zlabel('$z$', fontdict={'size': 14})
        ax.set_ylabel('$y$', fontdict={'size': 14})
        ax.set_xlabel('$x$', fontdict={'size': 14})
    else:
        assert False

    plt.legend()
    plt.show()



data = data_gen(100, 3)
pca_data, optimalVec, data_centralized, mean = PCA(data, 2)
PCA_figure(3, data, pca_data, mean, optimalVec)





