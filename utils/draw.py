import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import os

#直线方程函数
def f_1(x, A, B):
    return A*x + B

def scatter_plot(data, title=None, x_label="x_label", y_label="y_label", color_point="red", color_line="blue"):
    plt.figure()

    x0 = []
    y0 = []
    for _, i in data.items():
        x0.append(i[0])
        y0.append(i[1])

    #绘制散点
    plt.scatter(x0[:], y0[:], 25, color_point)

    #直线拟合与绘制
    A1, B1 = optimize.curve_fi