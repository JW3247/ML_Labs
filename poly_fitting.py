import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn import preprocessing

def training_data_gen(size):    #生成数据量为size的训练集
    i = 0
    x = []  #横坐标
    t = []  #利用sin算出的纵坐标
    for j in range(size):
        t.append(math.sin(2*math.pi*i))
        x.append(i)
        i = (j+1)/(size-1)
    #print(t)
    y = np.array(t)
    #print(y)
    noise = np.random.normal(0, 0.1, size)    #生成均值0的高斯噪声
    #print(noise)
    y = y + noise   #数据加噪声
    #print(y)
    y_col = y.reshape(size, 1)
    y_norm = preprocessing.MaxAbsScaler().fit_transform(y_col)  #归一
    y = y_norm.reshape(1, -1)
    #print(y)
    #plt.scatter(x, y)
    #plt.show()
    #x = np.array(x)

    return x, y


def get_result(x, y, m, size):  #最小二乘法求解析解
    X = np.zeros(shape=(size, m+1))
    for i in range(size):   #计算Xi，Xi各维是xi的幂
        Xi = []
        xi = x[i]
        for j in range(m+1):
            Xi.append(xi**j)
        X[i] = Xi
    X_transpose = np.transpose(X)
    w = np.dot(np.linalg.inv(np.dot(X_transpose, X)), np.dot(X_transpose, y.reshape(size, 1)))  #解w
    w = np.array(w)
    esti_x = np.arange(0, 1, 0.01)  #计算、绘制拟合函数的曲线
    esti_y = []
    for i in esti_x:
        esti_Xi = []
        for j in range(m+1):
            esti_Xi.append(i**j)
        res = (np.dot(np.transpose(w), esti_Xi))[0]
        esti_y.append(res)
    plt.subplot(2, 2, 1)
    plt.title('LSM')
    plt.scatter(x, y)
    plt.plot(esti_x, esti_y)
    #plt.show()
    
    return w


def get_result_with_punishment(x, y, m, size, par_lambda):  #最小二乘法求解解析解，带惩罚项
    X = np.zeros(shape=(size, m+1))
    punishment = np.zeros(shape=(m+1, m+1))
    for i in range(size):   #计算Xi,X
        Xi = []
        xi = x[i]
        for j in range(m+1):
            Xi.append(xi**j)
        X[i] = Xi
    actual_lambda = np.exp(par_lambda)
    for i in range(m+1):
        punishment[i][i] = actual_lambda
    X_transpose = np.transpose(X)
    w = np.dot(np.linalg.inv(np.dot(X_transpose, X)+punishment), np.dot(X_transpose, y.reshape(size, 1)))   #计算带惩罚项的解析解
    w = np.array(w)
    esti_x = np.arange(0, 1, 0.01)  #计算、绘制拟合曲线
    esti_y = []
    for i in esti_x:
        esti_Xi = []
        for j in range(m+1):
            esti_Xi.append(i**j)
        res = (np.dot(np.transpose(w), esti_Xi))[0]
        esti_y.append(res)
    plt.subplot(2, 2, 2)
    plt.title('LSM with punishment')
    plt.scatter(x, y)
    plt.plot(esti_x, esti_y)
    #plt.show()
    
    return w


def get_result_by_gradient_descent(x, y, m, size, eta): #梯度下降法求解，不带惩罚项
    w = []
    #eta = 0.03
    iter_times = 0
    iter = []
    loss_X = []
    y = y.reshape(size, 1)

    for i in range(m+1):
        w.append(0) 
    w = np.array(w).reshape(m+1, 1)
    X = np.zeros(shape=(size, m+1))
    for i in range(size):
        Xi = []
        xi = x[i]
        for j in range(m+1):
            Xi.append(xi**j)
        X[i] = Xi
    mid = np.dot(X, w) - y
    loss = np.dot(np.transpose(mid), mid)[0][0]
    while True: #迭代直到收敛
        iter_times = iter_times + 1
        iter.append(iter_times)
        loss_X.append(loss)   #计算损失函数

        grad = calculate_grad(X, y, w)
        w = w - eta*grad    #梯度更新
        mid = np.dot(X, w) - y
        loss_after = np.dot(np.transpose(mid), mid)[0][0]

        if converge1(loss, loss_after, grad)==1:    #收敛，跳出循环
            break
        loss = loss_after
    #print(grad)
    w = np.array(w)
    #print(iter_times)

    esti_x = np.arange(0, 1, 0.01)  #计算、绘制拟合曲线
    esti_y = []
    for i in esti_x:
        esti_Xi = []
        for j in range(m+1):
            esti_Xi.append(i**j)
        res = (np.dot(np.transpose(w), esti_Xi))[0]
        esti_y.append(res)
    plt.subplot(2, 2, 3)
    plt.title('Gradient descent')
    plt.scatter(x, y)
    plt.plot(esti_x, esti_y)
    #plt.show()
    #plt.plot(iter, loss)
    #plt.show()
    return w, iter, loss_X


def calculate_grad(X, y, w):    #计算梯度
     X_transpose = np.transpose(X)
     grad = np.dot(X_transpose, np.dot(X, w) - y)
     return grad

def converge1(loss, loss_after, grad):  #梯度下降法，判断是否收敛；当变化足够小且梯度模足够接近0，认为收敛
    len = np.linalg.norm(grad)
    if loss - loss_after>=1e-6:
        return 0
    if len>=1e-3:
        return 0
    return 1

def get_result_by_cg(x, y, m, size):    #共轭梯度法求解，不带惩罚项
    w = []
    for i in range(m+1):
        w.append(0)
    w = np.array(w).reshape(m+1, 1)
    X = np.zeros(shape=(size, m+1))
    y = y.reshape(size, 1)
    for i in range(size):
        Xi = []
        xi = x[i]
        for j in range(m+1):
            Xi.append(xi**j)
        X[i] = Xi
    Xt = np.transpose(X)
    A = np.dot(Xt, X)   #Ax=b，计算参数A，b
    b = np.dot(Xt, y)
    p = r = b - np.dot(A, w)    #初始方向、初始残差均为负梯度，w初始化全0
    while True:     #循环直到收敛
        tmp1 = np.dot(np.transpose(r), r)   #计算中间值
        tmp2 = np.dot(A, p)

        alpha = tmp1 / np.dot(np.transpose(p), tmp2)    #计算步长
        w = w + alpha*p     #更新参数
        r = r - alpha*tmp2  #更新残差
        p = r + (np.dot(np.transpose(r), r)/tmp1)*p     #更新优化方向
        if converge2(r, m)==1:
            break
    esti_x = np.arange(0, 1, 0.01)      #计算、绘制拟合曲线
    esti_y = []
    for i in esti_x:
        esti_Xi = []
        for j in range(m+1):
            esti_Xi.append(i**j)
        res = (np.dot(np.transpose(w), esti_Xi))[0]
        esti_y.append(res)
    plt.subplot(2, 2, 4)
    plt.title('CG')
    plt.scatter(x, y)
    plt.plot(esti_x, esti_y)
    #plt.show()
    #plt.plot(iter, loss)
    #plt.show()
    return w


    
def converge2(r, m):    #共轭梯度法，判断是否收敛；若梯度足够接近0，认为收敛
    for i in range(m+1):
        if(np.abs(r[i][0])>=1e-6):
            return 0
    return 1


data_size = int(input('Please input size of data: '))
m = int(input('Please input polynomial order: '))
par_lambda = int(input('Please input ln(lambda): '))
eta = float(input('Please input eta for gradient descent: '))
x, y = training_data_gen(data_size)
w0 = get_result(x, y, m, data_size)
w1 = get_result_with_punishment(x, y, m, data_size, par_lambda)
w2, iter, loss = get_result_by_gradient_descent(x, y, m, data_size, eta)
w3 = get_result_by_cg(x, y, m, data_size)
print('lsm: ', w0.reshape(1,-1))
print('lsm with punishment: ', w1.reshape(1,-1))
print('gradient descent: ', w2.reshape(1,-1))
print('conjugate gradient: ', w3.reshape(1,-1))
plt.tight_layout()
plt.show()
plt.plot(iter, loss)    #绘制梯度下降法的损失函数
plt.title('Loss function in GD process')
plt.show()



    