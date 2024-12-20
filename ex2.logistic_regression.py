'''Copyright 2024 Jiacheng Ni

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.'''
#!/usr/bin/env python
# coding: utf-8

# # 逻辑回归 Logistic Regression
# 
# 此Notebook是配合Andrew Ng "Machine Learning"中[逻辑回归](https://github.com/loveunk/machine-learning-deep-learning-notes/blob/master/machine-learning/logistic-regression.md)部分学习使用。
#
# 测试用python版本为3.6
# * 机器学习路径：https://github.com/loveunk/machine-learning-deep-learning-notes/
# * 内容正文综合参考网络资源，使用中如果有疑问请联络：www.kaikai.ai

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.metrics import classification_report  # 这个包是评价报告

# ## 准备数据

# 读取数据文件
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
data.head()

# 查看数据的描述性统计信息
data.describe()

# 在训练的初始阶段，我们将要构建一个逻辑回归模型来预测，某个学生是否被大学录取。
# 创建两个分数的散点图，并使用颜色编码来可视化，如果样本是正的（被接纳）或负的（未被接纳）。
positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

# 准备特征和标签
def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # 创建全1列
    data = pd.concat([ones, df], axis=1)  # 合并数据
    return data.iloc[:, :-1].values  # 返回特征矩阵

def get_y(df):
    return np.array(df.iloc[:, -1])  # 返回标签列

def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())  # 特征缩放

X = get_X(data)
y = get_y(data)

# 显示X和y的形状
print(X.shape, y.shape)

# ## Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 检查Sigmoid函数
nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
ax.set_xlabel('z', fontsize=18)
ax.set_ylabel('g(z)', fontsize=18)
ax.set_title('Sigmoid Function', fontsize=18)
plt.show()

# ## 代价函数
def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

# 初始化theta为0
theta = np.zeros(X.shape[1])

# 计算代价函数
print(f"Initial cost: {cost(theta, X, y)}")

# ## 梯度函数
def gradient(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

# 检查梯度
print("Gradient: ", gradient(theta, X, y))

# ## 使用SciPy优化器拟合参数
import scipy.optimize as opt

# 使用Newton-CG优化器来最小化代价函数，寻找最佳的theta
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
theta_opt = res.x
print(f"Optimized theta: {theta_opt}")

# 计算优化后的代价
print(f"Optimized cost: {cost(theta_opt, X, y)}")

# ## 用训练集做预测
def predict(theta, X):
    return sigmoid(X @ theta.T) >= 0.5  # 如果预测概率>=0.5，则预测为1，否则为0

# 使用优化后的theta做预测
predictions = predict(theta_opt, X)

# 计算训练集准确率
accuracy = np.mean(predictions == y) * 100
print(f"Training Accuracy: {accuracy:.2f}%")

# ## 使用分类报告进行评估
print("Classification Report:")
print(classification_report(y, predictions))

# ## 绘制决策边界
def plot_decision_boundary(theta, X, y):
    # 设置绘图网格的范围
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x2_min, x2_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))

    # 预测网格中的每个点
    Z = predict(theta, np.column_stack([np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()]))
    Z = Z.reshape(xx1.shape)

    # 绘制决策边界
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 1], X[:, 2], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# 绘制决策边界
plot_decision_boundary(theta_opt, X, y)
