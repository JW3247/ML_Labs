import os
import math
import PIL
import numpy as np
import matplotlib.pyplot as plt

def PCA(data, m):
    #print(data.shape)
    N = data.shape[1]
    d = data.shape[0]
    mean = np.mean(data, axis=1) #compute row mean

    data_centralized = np.zeros(data.shape) #d*N
    for i in range(d):  #去中心化
        data_centralized[i] = data[i] - mean[i]

    cov = np.dot(data_centralized, np.transpose(data_centralized))  #协方差矩阵

    eigenvalues, feature_vectors = np.linalg.eig(cov)  # 特征值，特征向量
    sortedIndex = np.argsort(eigenvalues)   #排序
    optimalVec = np.zeros((d, m), dtype='complex_')
    for j in range(m):  #取最大的m个特征值对应的特征向量
        optimalVec[:,j]=feature_vectors[:,(sortedIndex[d-j-1])]
    return optimalVec, mean

def readpicture(facedataset):
    result = []
    for i in range(30): #读入30个数据
        filename = facedataset[i]
        im = PIL.Image.open("./facedata/" + filename)
        x = np.array(im).tolist()   #得到一个list嵌套
        y = []
        #按行拼接数据
        for j in range(50):
            y = y + x[j]
        result.append(y)
    return np.array(result) #30*2500

def facePCA():   
    facedatas = []    #文件路径
    for root, dirs, files in os.walk("./facedata"):
            facedatas = files
    faceresult = readpicture(facedatas)
    dim = [30, 20, 10, 5, 2]
    for i in range(5):
        optimalVec, mean = PCA(faceresult.T, dim[i])
        face_pca = (optimalVec.dot(((faceresult - mean).dot(optimalVec)).T)).T + mean    #重构数据
        for j in range(15):
            snr = psnr(faceresult[j].T, face_pca[j])    #单个图像信噪比
            res = face_pca[j].reshape(50,50)    #重构图像
            res = np.array(res, dtype = np.int32)
            plt.subplot(3, 5, j+1)  #画图
            plt.imshow(res, cmap = plt.cm.gray)
            plt.title('m=' + str(dim[i]) + ', SNR=' + str("%.2f" % snr))
        plt.tight_layout()
        plt.show()

def psnr(img1Data, img2Data):   #计算信噪比
    mse = np.mean((img1Data / 255. - img2Data / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # 使用的信噪比公式为20 log_10^(MAX/sqrt(MSE))
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

facePCA()
