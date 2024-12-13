import numpy as np

# 定义所需的变量
numpoints = 1000  # 数据点的数量

# 重新定义x坐标的范围为0到0.06
x = np.linspace(0, 1, numpoints)
c=60
# 计算上表面和下表面的y坐标
yupper = np.array([0.5 * (0.2969 * np.sqrt(x[i]) - 0.1260 * x[i] - 0.3516 * x[i]**2 + 0.2843 * x[i]**3 - 0.1015 * x[i]**4) for i in range(numpoints)])
ylower = -yupper

# 将数据保存到.txt文件，格式为：x坐标, 上表面y坐标, 下表面y坐标
with open('naca.txt', 'w') as file:
    for i in range(numpoints):
        file.write(f"{x[i]*c} {yupper[i]*c} {0}\n")
    for i in reversed(range(numpoints)):     

        file.write(f"{x[i]*c} {ylower[i]*c} {0}\n")

with open('naca.txt', 'r') as file:
    naca_data_full = file.readlines()

print(naca_data_full)



