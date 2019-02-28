#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy
import matplotlib.pyplot as plt

#作者:神仙

sample_x = None     #x样本矩阵
sample_y = None     #y样本矩阵
sample_num = []     #数据的数量（矩阵的维度）
theta1 = None       #第一层权值
theta2 = None       #第二层权值
b1 = None           #第一层偏置项
b2 = None           #第二层偏执项
linenum = 3         #第二层神经元个数
alpha = 0.003       #学习率

#读取数据和初始化数据
def init_sample(filename):
    global sample_x
    global sample_y
    global sample_num
    global theta1
    global theta2
    global b1
    global b2

    temp_x = []
    temp_y = []
    if not isinstance(filename,str):
        raise ValueError('parameter filename is a str type,please check your parameter type')
    with open(filename,'r') as f:
        for line in f.readlines():
            temp = line.split(':')
            temp_x.append(list(map(float,temp[0].split(","))))
            temp_y.append(float(temp[1].strip()))
    sample_num = [len(temp_y),len(temp_x[0])]
    sample_x = numpy.array(temp_x).reshape(sample_num[0],sample_num[1])
    sample_y = numpy.array(temp_y).reshape(sample_num[0],1)
    theta1 = numpy.random.rand(sample_num[1],linenum).reshape(sample_num[1],linenum)
    theta2 = numpy.random.rand(linenum,1).reshape(linenum,1)
    b1 = numpy.random.rand(sample_num[1]+1,1).reshape(1,sample_num[1]+1)
    b2 = numpy.array([1])

#sigmoid函数
def sigmoid(z):
    return 1/(1+numpy.exp(-z))

#特征缩放
def scalling(x):
    max = numpy.max(x,0)
    min = numpy.min(x,0)
    return (x-min)/(max-min),max,min

#前向传播
def forward_propagation(X,theta1,theta2,b1,b2):
    z1 = X@theta1+b1
    a1 = sigmoid(z1)

    z2 = a1@theta2+b2
    a2 = sigmoid(z2)

    return z1,a1,z2,a2


#反向传播
def back_propagation(*r):
    global theta1
    global theta2
    global b1
    global b2
    global alpha

    z1,a1,z2,a2 = r[0]
    a0 = sample_x.copy()
    d2 = a2-sample_y
    d1 = d2.dot(theta2.T)*(a1*(1-a1))

    loss = -(numpy.sum(sample_y*numpy.log(a2)+(1-sample_y)*numpy.log(1-a2)))/sample_num[0]

    dtheta1 = a0.T@(d1)
    db1 = numpy.sum(d1,0)
    dtheta2 = a1.T@(d2)
    db2 = numpy.sum(d2,0)

    theta1 = theta1 - alpha*dtheta1
    theta2 = theta2 - alpha*dtheta2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2

    return loss

#数据可视化
def data_show(x,y,plot_max,plot_min):
    y = numpy.mat(y)
    x = numpy.mat(x)

    plt.scatter(x[:, 0][y == 0].A, x[:, 1][y == 0].A, marker='x', label='0')
    plt.scatter(x[:, 0][y == 1].A, x[:, 1][y == 1].A, marker='o', label='1')
    plt.grid()
    plt.legend()

    plotx = numpy.arange(0,100,0.1)
    ploty = numpy.arange(0, 100, 0.1)
    plotx, ploty = numpy.meshgrid(plotx, ploty)
    plot_new = numpy.c_[plotx.ravel(),ploty.ravel()]
    plot_s = (plot_new-plot_min)/(plot_max-plot_min)

    plot_z1 = plot_s@theta1+b1
    plot_a1 = sigmoid(plot_z1)
    plot_z2 = plot_a1@theta2+b2
    plot_a2 = sigmoid(plot_z2)

    plot_new = numpy.reshape(plot_a2,plotx.shape)

    plt.contourf(plotx,ploty,plot_new,1,alpha=0.5)

    plt.show()

if __name__ == '__main__':
    loss = 1
    i = 0
    init_sample('sample.txt')
    temp_x = sample_x.copy()
    temp_y = sample_y.copy()
    sample_x,max,min = scalling(sample_x)
    #自动下降
    while True:
        temp = loss
        result = forward_propagation(sample_x,theta1,theta2,b1,b2)
        loss = back_propagation(result)
        i +=1
        if i%500 == 0:
            print('loss',loss)
        if abs(loss-temp)<0.000001 and loss<0.001:
            break

    print('theta1',theta1)
    print('theta2',theta2)
    print('b1',b1)
    print('b2',b2)
    data_show(temp_x,temp_y,max, min)
