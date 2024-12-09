import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Ellipse

def data_gen(mean, number, k): #k_means的数据生成函数
    cov = [[0.3,0],[0,0.3]] #2维高斯的协方差矩阵
    data = []
    for i in range(k):  #k类
        ith_mean = [mean[i][0], mean[i][1]]
        for n in range(number[i]):
            data.append(np.random.multivariate_normal(ith_mean, cov).tolist()) #生成数据
    return np.array(data)


def k_Means(data, k):   #k_means聚类过程
    
    row = data.shape[0] #data:row*col矩阵
    col = data.shape[1]
    center = np.zeros((k, col))   #k个1行2列
    tag = np.zeros((row, ), dtype=int)  #样本标记
    for i in range(k):  #从数据中随机选k个样本点作初始中心
        center[i, :] = data[np.random.randint(0, high = row), :]
    times = 0
    while True:
        times += 1
        #print(times)
        distance = np.zeros(k)
        prev_tag = tag
        for p in range(row):
            for q in range(k):  #计算数据点与k个中心的距离
                distance[q] = np.linalg.norm(data[p, :]-center[q, :]) 
        #print(distance)
            index = np.argmin(distance) #选择最小距离对应的类别，贴标签
        #print(index)
            tag[p] = index
        #print(prev_tag)
        #print(tag)
        
        count  = np.zeros(k)
        new_center = np.zeros((k, col))
        for i in range(row):
            tag_i = tag[i]
            count[tag_i] = count[tag_i] + 1 #计每个类别各自的总数
            new_center[tag_i, :] += data[i, :] #各类数据求和
        #print(count)
        for i in range(k):  #每类新中心=该类数据点均值
            new_center[i, :] = new_center[i, :] / count[i]

        if np.linalg.norm(new_center - center) < 1e-5:  #中心不再变化，则认为收敛
            break
        else:
            center = new_center
        
    return center, tag


def evaluate(data, k, miu_list, sigma_list, pi_list):   #EM-E
    N = data.shape[0]
    gamma_z = np.zeros((N, k))
    tag = np.zeros(N, dtype=int)
    for n in range(N):
        denom = 0
        pi_times_px = np.zeros(k)   #先验*正态
        for i in range(k):
            pi_times_px[i] = pi_list[i] * st.multivariate_normal.pdf(data[n], mean = miu_list[i], cov = sigma_list[i]) 
            denom += pi_times_px[i]
        for j in range(k):  #第n个样本来自第j类的概率-贝叶斯公式
            gamma_z[n][j] = pi_times_px[j] / denom
    c = [[]for i in range(k)]
    for n in range(N):
        index = np.argmax(gamma_z[n])   #选择最大先验的高斯，贴标签
        c[index].append(data[n].tolist())
        tag[n] = index
    return gamma_z, tag

def maxmize(data, k, miu_list, gamma_z):    #EM-M
    Nk = np.zeros(k)    #各类的有效样本数
    N = data.shape[0]
    col = data.shape[1]
    new_miu_list = np.zeros(miu_list.shape) #k*2
    new_sigma_list = np.zeros((k, col, col))
    #new_pi_list = np.zeros(k)
    for i in range(k):
        for n in range(N):
            Nk[i] += gamma_z[n][i]  #第i类数据个数=每个数据来自第i类的概率之和
    
    new_pi_list = Nk / N    #更新先验
    
    for j in range(k):
        gamma = gamma_z[:, j]
        gamma2 = gamma
        for i in range(col-1):
            gamma2 = np.column_stack((gamma2, gamma))
        gamma_z_jk = gamma_z[:, j].reshape(N, 1) 

        new_miu_list[j] = np.dot(np.transpose(gamma_z_jk), data) / Nk[j]    #更新均值
        new_sigma_list[j] = np.dot(np.transpose(data-miu_list[j]), np.multiply(data-miu_list[j], gamma2)) / Nk[j] #gamma_z为数组
    
    return new_miu_list, new_sigma_list, new_pi_list

def log_likelihood(data, k, miu_list, sigma_list, pi_list): #EM的对数似然函数
    N = data.shape[0]
    l = 0
    sum = 0
    for i in range(N):
        for j in range(k):
            sum += pi_list[j] * st.multivariate_normal.pdf(data[i], mean=miu_list[j], cov=sigma_list[j])
        l += np.log(sum)
    return l


def EM(data, k):
    k_center, k_tag = k_Means(data, k) 
    row = data.shape[0]
    col = data.shape[1]
    index = np.random.randint(0, row)
    miu_list = k_center
    '''
    miu_list = np.zeros((k, col)) #k*2
    miu_list[0] = data[index]   #随机选一个数据点作一个初始均值
    for i in range(k-1):    #再选k-1个数据点作初始均值，其中每一个与现有均值点的距离之和最大
        distanceSum = []
        for n in range(row):
            distanceSum.append(np.sum([np.linalg.norm(data[n, :]-miu_list[j, :]) for j in range(len(miu_list))] ))
        miu_list[i+1] = data[np.argmax(distanceSum)]
    '''
    sigma_list = np.zeros((k, col, col))
    for i in range(k):  #初始协方差矩阵初始化为对角线元全为0.1的对角阵
        sigma_list[i] = np.eye(col, dtype=float) * 0.1

    pi_list = np.ones(k) * (1.0/k)  #先验全初始化为1/k
    l = log_likelihood(data, k, miu_list, sigma_list, pi_list)
    
    while True:
        gamma_z, tag = evaluate(data, k, miu_list, sigma_list, pi_list)
        miu_list, sigma_list, pi_list = maxmize(data, k, miu_list, gamma_z)
        #print(sigma_list)
        new_l = log_likelihood(data, k, miu_list, sigma_list, pi_list)
        if new_l - l<1e-6:  #对数似然不再变化，认为收敛
            break
        l = new_l

    return miu_list, sigma_list, pi_list, tag

def GMMdata():  #GMM数据生成
    # 第一簇的数据
    num1, mu1, var1 = 200, [8, 10], [1, 3]
    #num2, mu2, var2 = 300, [8, 2], [2, 2]
    #num3, mu3, var3 = 400, [2, 6], [1, 2]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 300, [8, 2], [2, 2]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 400, [2, 6], [1, 2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    true_mu = [mu1, mu2, mu3]
    true_Var = [var1, var2, var3]
    return X, X1, X2, X3, true_mu, true_Var


def KMeansImage(k, center, cluster):    #kMeans聚类画图
    #fig, axes = plt.subplots(1, 1)
    plt.plot()
    plt.title("K-Means", fontsize=16)
    col = cluster.shape[1]
    c = ["#00CED1", "#DC143C", "#7CFC00", "#4B0082", "#EE82EE" ]
    for n in range(cluster.shape[0]):
        for i in range(k):
            if cluster[n][col-1] == i:
                plt.scatter(cluster[n][0], cluster[n][1], c = c[i], marker="x")
    plt.scatter(center[:, 0], center[:, 1], facecolor="red", edgecolor="black", label="center")
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    plt.legend()
    plt.show()


def GMMImage(Mu, Var, X1, X2, X3, Mu_true=None, Var_true=None): #GMM画图
    n_clusters = len(Mu)
    ax = plt.gca()
    # 画数据点
    ax.scatter(X1[:, 0], X1[:, 1], marker="x", label="class 1")
    ax.scatter(X2[:, 0], X2[:, 1], marker="x", label="class 2")
    ax.scatter(X3[:, 0], X3[:, 1], marker="x", label="class 3")
    # 画中心点
    ax.scatter(Mu[:, 0], Mu[:, 1], facecolor="red", edgecolor="black", label="center")
    
    # 画GMM学习出的高斯椭圆
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'ls': ':', 'edgecolor': "red"}
        miu = []
        miu.append(Mu[i][0])
        miu.append(Mu[i][1])
        ellipse = Ellipse(miu, 3 * Var[i][0][0], 3 * Var[i][1][1], **plot_args)
        ax.add_patch(ellipse)
    
 
    # 画真实的高斯椭圆
    if (Mu_true is not None) & (Var_true is not None):
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'alpha': 0.5, 'edgecolor': "purple"}
            #print(Mu_true[i])
            #print(Var_true[i][0])
            ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
            ax.add_patch(ellipse)
    
    ax.set_title("GMM", fontsize=16)
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$y$", fontsize=14)
    ax.legend()
    plt.show()


def UCIiris():  #UCIiris数据集聚类
    k = 3
    data = pd.read_table('iris.data', sep=',', names=['x1', 'x2', 'x3', 'x4', 'y']) #读取数据
    x = data.drop('y', axis=1)  #去掉y列
    y = data['y']
    X = np.array(x)
    Y = np.array(y)
    size = len(Y)
    miu_list, sigma_list, pi_list, tag = EM(X, k)   #EM算法求参数
    print(miu_list)
    print(sigma_list)
    print(pi_list)
    print(tag)
    return miu_list, sigma_list, pi_list, tag


'''
k = 3
means = [[3, 1], [-3, 1], [0, 4]]
num = [100, 100, 200]
data = data_gen(means, num, k)
'''
k = 5
means = [[1, 0], [-1, 0], [0, 1], [1, 3], [-1, 3]]
num = [100, 100, 100, 100, 100]
data = data_gen(means, num, k)
center, tag = k_Means(data, k)
print(center)
print(tag)
cluster = np.column_stack((data, np.transpose(tag)))
KMeansImage(k, center, cluster)
'''
k = 3
X, X1, X2, X3, true_mu, true_Var = GMMdata()
miu_list, sigma_list, pi_list, c = EM(X, k)

print(miu_list)
print(sigma_list)
print(pi_list)

GMMImage(miu_list, sigma_list, X1, X2, X3, true_mu, true_Var)

#UCIiris()
'''