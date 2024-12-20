#!/usr/bin/env python
# coding: utf-8

# # 一个完整的scipy.optimize.minimize训练例子
# 
# 测试用python版本为3.6
# * 机器学习路径：https://github.com/loveunk/machine-learning-deep-learning-notes/
# * 内容正文综合参考网络资源，使用中如果有疑问请联络：www.kaikai.ai

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

# 定义训练数据集 X 和标签 y
X_train = np.array([[1.,  1.],
                    [1.,  2.],
                    [-1., -1.],
                    [-1., -2.]])
y_train = np.array([1, 1, 0, 0])

# 对 X 数据添加 x_0 = 1 (偏置项)
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Sigmoid 函数实现
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 计算代价函数 (Cost Function)
def cost(theta, X, y):
    first = - y.T @ np.log(sigmoid(X @ theta))
    second = (1 - y.T) @ np.log(1 - sigmoid(X @ theta))
    return ((first - second) / len(X)).item()

# 预测函数
def hypothesis(X, theta):
    return sigmoid(X @ theta)

# 用于 `minimize` 函数的代价函数包装器
def cost_wrapper(theta):
    return cost(theta, X_train, y_train)

# 预测函数包装器
def hypothesis_wrapper(theta):
    return hypothesis(X_train, theta)

# 计算梯度
def gradient(theta):
    return (1 / X_train.shape[0]) * ((hypothesis_wrapper(theta) - y_train).T @ X_train)

# 初始化theta为 [1, 1., 2.]
theta_train = np.array([1, 1., 2.])

# 使用CG（共轭梯度）方法最小化代价函数，获得最优的theta
theta_opt = optimize.minimize(cost_wrapper, theta_train, method='CG', jac=gradient)
print(theta_opt)

# 用于预测的网格数据
delta = 0.2
px = np.arange(-3.0, 3.0, delta)
py = np.arange(-3.0, 3.0, delta)
px, py = np.meshgrid(px, py)
px = px.reshape((px.size, 1))
py = py.reshape((py.size, 1))
pz = np.hstack((np.hstack((np.ones((px.size, 1)), px)), py))

# 3D 绘图展示训练数据、预测数据
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制训练数据点（红色）
ax.scatter(X_train[:, 1], X_train[:, 2], y_train, color='red', marker='^', s=200, label='Training Data')

# 绘制预测数据（灰色），用于展示模型的分类
ax.scatter(px, py, (hypothesis(pz, theta_opt.x)), color='gray', label='Prediction Data')

# 添加图例
ax.legend(loc=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Classification')

# 显示图形
plt.show()
