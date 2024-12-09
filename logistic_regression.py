import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import split
import pandas as pd
from numpy.core.fromnumeric import shape, size, transpose
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(index): #按指数正负选择不同的公式，避免exp溢出
    if index>=0:
        return 1.0/(1+np.exp(-index))
    else:
        return np.exp(index)/(1+np.exp(index))


def calc_loss(X, Y, w, size):   #计算损失函数
    loss = 0
    for i in range(size):
        index = np.dot(X[i], w)[0]
        if Y[i]==1:
            loss = loss + np.log(sigmoid(index)+1e-6)   #正例，传参为index，1e-6调整精度
        else:
            loss = loss + np.log(sigmoid(-index)+1e-6)  #反例，传参为-index，1e-6调整精度
    return -loss    #损失为似然取负


def gradient_descent(X, Y, size, n):    #梯度下降法，无惩罚项
    w = []
    loss_x = []
    loss_y = []
    iter_times = 0
    eta = 0.01
    for i in range(n):
        w.append(0) 
    w = np.array(w).reshape(n, 1)
    loss = calc_loss(X, Y, w, size)
    while True: #迭代直到收敛
        iter_times = iter_times + 1
        loss_x.append(iter_times)
        loss_y.append(loss)
        #print(loss)
        grad = calculate_grad(X, Y, w, size, n)
        #print(grad)
        w = w - eta*grad    #梯度更新
        loss_after = calc_loss(X, Y, w, size)
        #print(loss_after)
        if converge1(loss, loss_after, grad)==1:    #收敛，跳出循环
            break
        loss = loss_after

    print(grad)
    print(iter_times)
    w = np.array(w)
    return w

def gradient_descent_with_regularization(X, Y, size, n):    #梯度下降法，有正则项
    w = []
    loss_x = []
    loss_y = []
    iter_times = 0
    eta = 0.01  #步长
    param_lambda = np.exp(-5)   #惩罚项比例
    for i in range(n):
        w.append(0) 
    w = np.array(w).reshape(n, 1)
    loss = calc_loss(X, Y, w, size)
    while True: #迭代直到收敛
        iter_times = iter_times + 1
        loss_x.append(iter_times)
        loss_y.append(loss)
        #print(loss)
        grad = calculate_grad(X, Y, w, size, n)
        w = w - eta*param_lambda*w - eta*grad    #梯度更新
        loss_after = calc_loss(X, Y, w, size)
        #print(loss_after)
        if converge1(loss, loss_after, grad)==1:    #收敛，跳出循环
            break
        loss = loss_after

    print(grad)
    print(iter_times)
    w = np.array(w)
    return w

def converge1(loss, loss_after, grad):  #梯度下降法，判断是否收敛；当损失变化足够小，认为收敛
    len = np.linalg.norm(grad)
    #if len>=1e-3:
        #return 0
    if loss - loss_after >= 1e-6:
        return 0
    return 1


def calculate_grad(X, Y, w, size, n):    #计算梯度
    grad = []
    for i in range(n):
        grad.append(0) 
    grad = np.array(grad).reshape(n, 1)
    for i in range(size):   #此处计算的是l(w)的梯度
         Xi = np.array(X[i])
         Xi_T = Xi.reshape(n,1)
         index = np.dot(Xi, w)[0]
         coefficent = Y[i] - sigmoid(index)
         grad = grad + coefficent * Xi_T
    return -grad    #最终结果取负



def data_generator(size, posRate, naive):   #数据生成
    pos = int(np.ceil(size*posRate))
    neg = size - pos
    miu1 = [1, 1]
    miu2 = [-1, -1]
    cov1 = [[0.5, 0], [0, 0.3]] #满足朴素贝叶斯
    cov2 = [[0.5, 0.1], [0.1, 0.3]] #不满足朴素贝叶斯
    X = np.zeros((size, 3))
    if naive==1:    #满足朴素贝叶斯
        X[:pos, 1:] = np.random.multivariate_normal(miu1, cov1, size = pos)
        X[pos:, 1:] = np.random.multivariate_normal(miu2, cov1, size = neg)
    else:   #不满足
        X[:pos, 1:] = np.random.multivariate_normal(miu1, cov2, size = pos)
        X[pos:, 1:] = np.random.multivariate_normal(miu2, cov2, size = neg)

    X[:, 0:1] = 1   #增广，加一列1
    Y = np.zeros(size)
    Y[:pos] = 1
    Y[pos:] = 0
    return X, Y, pos

def calculate_accuracy(X, Y, w, size):  #计算精度
    err_count = 0
    for i in range(size):
        x = X[i][1]
        y = X[i][2]
        predict_y = -(w[0][0]+x*w[1][0])/w[2][0]
        if Y[i]==1 and y<predict_y: #正例，分成反例
            err_count = err_count + 1
        elif Y[i]==0 and y>predict_y:   #反例，分成正例
            err_count = err_count + 1
    return 1 - err_count/size


def figure(X, pos, verify_X, v_pos, w1, w2, acc1, acc2, acc3, acc4):    #画图
    x1 = -4
    y1 = -(w1[0][0]-4*w1[1][0])/w1[2][0]
    y3 = -(w2[0][0]-4*w2[1][0])/w2[2][0]
    x2 = 4
    y2 = -(w1[0][0]+4*w1[1][0])/w1[2][0]
    y4 = -(w2[0][0]+4*w2[1][0])/w2[2][0]
    X_pos = np.transpose(X[:pos, 1:])
    X_neg = np.transpose(X[pos:, 1:])
    vX_pos = np.transpose(verify_X[:v_pos, 1:])
    vX_neg = np.transpose(verify_X[v_pos:, 1:])
    plt.subplot(2, 2, 1)
    plt.title('Training set, accuracy: '+str(acc1))
    plt.scatter(X_pos[0], X_pos[1], c = '#00CED1', alpha = 0.4)
    plt.scatter(X_neg[0], X_neg[1], c = '#DC143C', alpha = 0.4)
    plt.plot([x1, x2], [y1, y2], linewidth = '1', c = '#00008B')
    plt.subplot(2, 2, 2)
    plt.title('Training set with reg, accuracy: '+str(acc2))
    plt.scatter(X_pos[0], X_pos[1], c = '#00CED1', alpha = 0.4)
    plt.scatter(X_neg[0], X_neg[1], c = '#DC143C', alpha = 0.4)
    plt.plot([x1, x2], [y3, y4], linewidth = '1', c = '#9400D3')
    plt.subplot(2, 2, 3)
    plt.title('Test set, accuracy: '+str(acc3))
    plt.scatter(vX_pos[0], vX_pos[1], c = '#00CED1', alpha = 0.4)
    plt.scatter(vX_neg[0], vX_neg[1], c = '#DC143C', alpha = 0.4)
    plt.plot([x1, x2], [y1, y2], linewidth = '1', c = '#00008B')
    plt.subplot(2, 2, 4)
    plt.title('Test set with reg, accuracy: '+str(acc4))
    plt.scatter(vX_pos[0], vX_pos[1], c = '#00CED1', alpha = 0.4)
    plt.scatter(vX_neg[0], vX_neg[1], c = '#DC143C', alpha = 0.4)
    plt.plot([x1, x2], [y3, y4], linewidth = '1', c = '#9400D3')
    plt.tight_layout()
    plt.show()


def readFromFile(rate): #从文件读取数据集
    data = pd.read_table('Skin_NonSkin.txt', sep='\t', names=['x1', 'x2', 'x3', 'y']) 
    x = data.drop('y', axis=1)  #去掉y列
    y = data['y']
    X = np.array(x)
    Y = np.array(y)
    size = len(Y)
    posNumber = 0
    for i in range(size):
        if Y[i] == 1:
            posNumber += 1
    
    pos = int(math.ceil(posNumber * rate))  #计算使用的正反例数量
    neg = int(math.ceil((size-posNumber) * rate))
    total = pos + neg

    slice = np.zeros(size).astype(np.int32)  #切片
    slice[:pos] = 1
    slice[posNumber:(posNumber + neg)] = 1
    X = X[slice == 1]*0.01  #对数据预处理
    Y = Y[slice == 1]
    for i in range(total):
        if Y[i]==2:
            Y[i] = 0

    ones_column = np.transpose(np.array(np.repeat(1, total)))   #增广X
    X = np.column_stack((ones_column, X))
    return X, Y, total

def calculate_accuracy_SS(X, Y, w, size):   #算Skin数据集的准确度
    err_count = 0
    for i in range(size):
        x1 = X[i][1]
        x2 = X[i][2]
        x3 = X[i][3]
        predict_y = -(w[0][0]+x1*w[1][0]+x2*w[2][0])/w[3][0]    #预测值
        if Y[i]==0 and x3<predict_y: #反例，分成正例
            err_count = err_count + 1
        elif Y[i]==1 and x3>predict_y:  #正例，分成反例
            err_count = err_count + 1
    return 1 - err_count/size

def figure_UCI(X, Y, w):    #画3D图
    ax = Axes3D(plt.figure())
    ax.scatter(X[:, 1], X[:, 2], X[:, 3], c=Y, cmap = 'coolwarm')
    x1 = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 1)
    x2 = np.arange(np.min(X[:, 2]), np.max(X[:, 2]), 1)
    x1, x2 = np.meshgrid(x1, x2)
    x3 = -(w[0][0] + w[1][0]*x1 + w[2][0]*x2)/w[3][0]
    ax.plot_surface(x1, x2, x3, rstride = 1, cstride = 1) 
    ax.set_zlim3d(np.min(x3)-1, np.max(x3)+1)
    plt.show()


'''
size = int(input('Please input size of data: '))
posRate = float(input('Please input rate of positive examples: '))
naive = int(input('Please decide whether the data satisfy the Bayes condition(1 for yes, 0 for no): '))
X, Y, pos = data_generator(size, posRate, naive)

w1 = gradient_descent(X, Y, size, 3)
w2 = gradient_descent_with_regularization(X, Y, size, 3)
print(w1)
print(w2)
verify_X, verify_Y, v_pos = data_generator(500, posRate, naive)
acc1 = calculate_accuracy(X, Y, w1, size)
acc2 = calculate_accuracy(X, Y, w2, size)
acc3 = calculate_accuracy(verify_X, verify_Y, w1, 500)
acc4 = calculate_accuracy(verify_X, verify_Y, w2, 500)
figure(X, pos, verify_X, v_pos, w1, w2, acc1, acc2, acc3, acc4)


'''
X, Y , size1 = readFromFile(0.0004)
print(size1)
testX, testY, size2 = readFromFile(0.002)
w1 = gradient_descent(X, Y, size1, 4)
w2 = gradient_descent_with_regularization(X, Y, size1, 4)
print(calculate_accuracy_SS(X, Y, w1, size1))
print(calculate_accuracy_SS(X, Y, w2, size1))
print(calculate_accuracy_SS(testX, testY, w1, size2))
print(calculate_accuracy_SS(testX, testY, w2, size2))
figure_UCI(X, Y, w1)
figure_UCI(X, Y, w2)
figure_UCI(testX, testY, w1)
figure_UCI(testX, testY, w2)
