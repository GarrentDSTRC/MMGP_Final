import numpy as np
import os
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

controlFre = 3000
c=0.1
U=0.1
mode="experiment_cluster"
def generate_waveform( X, folder_name,mode="CFD"):
    # 创建文件夹（如果不存在）
    #St, amplitude2, amplitude, phase_difference, alpha, alpha2=X
    St,  amplitude,  alpha=X
    phase_difference=0
    amplitude2=0
    alpha2=0
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if mode=="CFD":
        f=St
    else:
        f = X[0] * U / c
    T = 1 / f
    #print(T)
    points = T * controlFre


    # y(t)定义
    def y(t):
        return np.sin(t)

    # φ(t')定义
    def phi(t):
        #return t + alpha * np.sin(t) ** 2
        # return t - alpha * np.sin(0.5*t)**2
        return t - alpha * np.sin(0.5 * (t+np.pi/2)) ** 2

    # 使用数值方法找到φ^-1(t')的对应t值
    def phi_inverse(phi_prime, t_values):
        # 计算每一个t值对应的φ(t)和给定的t'之间的差异
        diffs = phi(t_values) - phi_prime
        # 找到差异最小的t值
        idx = np.argmin(np.abs(diffs))
        return t_values[idx]

    # 计算z(t')
    def z(phi_prime, t_values):
        t = phi_inverse(phi_prime, t_values)
        return y(t)

    # 主程序

    t_values = np.linspace(0, 2 * np.pi, int(points))
    phi_values = phi(t_values)

    # 计算z(t')的值
    z_values = [(amplitude * np.pi / 180) * z(phi_prime, t_values) for phi_prime in phi_values]
    # 使用线性插值生成均匀的时间点
    phi_uniform = t_values
    f_interp = interp1d(phi_values, z_values, fill_value='extrapolate')
    z_uniform = f_interp(phi_uniform)



    alpha=alpha2
    # 生成第二个波形
    phi_values2 = phi(t_values)
    z_values2 = [(amplitude2) * z(phi_prime, t_values) for phi_prime in phi_values2]

    # 使用线性插值生成均匀的时间点
    phi_uniform2 =t_values
    f_interp2 = interp1d(phi_values2, z_values2)
    z_uniform2 = f_interp2(phi_uniform2)


    # 找到最接近零的点的索引
    k = np.argmin(np.abs(z_uniform))
    # 计算滚动次数（左移k步）
    num_rolls = -k+int(-phase_difference/360  * len(z_uniform))
    z_uniform = np.roll(z_uniform, num_rolls)


    k = np.argmin(np.abs(z_uniform2))
    z_uniform2 = np.roll(z_uniform2, -k)

   # 保存第一个波形到文件

    with open(os.path.join(folder_name, "control.txt"), "w") as f:
        for value in z_uniform:
                f.write(str(value*180/np.pi) + "\n")
    # 保存第二个波形到文件
    with open(os.path.join(folder_name, "control2.txt"), "w") as f2:
        if mode == "CFD":
            for value in z_uniform2:
                f2.write(str(value) + "\n")
        else:
            for value in z_uniform2:
                f2.write(str(value) + "\n")


    # 绘制第一个波形
    plt.plot(phi_uniform, z_uniform, color="green", label="Pitching")
    # 绘制第二个波形
    plt.plot(phi_uniform2, z_uniform2, color="blue", label="Heaving")
    plt.xlabel("φ")
    plt.ylabel("z(φ)")
    plt.legend()
    #plt.show()
    plt.savefig("waveform.png")

    return f"Waveforms saved to {folder_name}/control.txt and {folder_name}/control2.txt"

# UPB=[0.9/0.4*0.06, 0.08/0.06, 85, -45, 0.9,0.9]
# LOWB=[0.4/0.4*0.06, 0.04/0.06, 55, -140, -0.9,-0.9]
#
# UPB=[0.9, 0.08, 85, -45, 0.9,0.9]
# LOWB=[0.4, 0.04, 55, -140, -0.9,-0.9]
UPB=[0.3, 85,0.9,9,9,35]
LOWB=[0.1, 15,-0.9,0,0,10]


# UPB=[0.4, 65,0.9,9,9,35,10]
# LOWB=[0.5, 0,-0.9,-9,0,10,1] #测试

UPB=[0.6, 75,0.9,9,9,35,10]
LOWB=[0.2, 15,-0.9,-9,0,10,1]
import torch
class Normalizer:
    def __init__(self, low_bound=LOWB, up_bound=UPB):
        self.low_bound = torch.tensor(low_bound, dtype=torch.float32)
        self.up_bound = torch.tensor(up_bound, dtype=torch.float32)

    def normalize(self, x):
        x=torch.as_tensor(x)
        return (x - self.low_bound) / (self.up_bound - self.low_bound)

    def denormalize(self, norm_x):
        norm_x = torch.as_tensor(norm_x)
        return norm_x * (self.up_bound - self.low_bound) + self.low_bound
norm=Normalizer()

x=[1,0	,0.1	,3.10E-01	,1.90E-01	,5.40E-01,0]
X=norm.denormalize(x).tolist()
print(X)


last_col = X[-1]  # Extract the last column
j=0
np.savetxt(r'.\MMGP_OL%d\dataX.txt' % (j % 8), np.array([[0, 0, 0, 0, X[-3],X[-2], X[-1], 6000]]),
                       delimiter=',', fmt='%d')
generate_waveform(X[0:3],"MMGP_OL%d"% (j % 8),mode=mode)
# Test
