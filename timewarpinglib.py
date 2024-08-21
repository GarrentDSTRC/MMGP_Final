import numpy as np
import os
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

def generate_waveform(amplitude, frequency, alpha, folder_name):
    # 创建文件夹（如果不存在）
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 将频率转化为周期
    T = 1 / frequency

    # y(t)定义
    def y(t):
        return  np.sin( t)

    # φ(t')定义
    def phi(t):
        return t + alpha * y(t) ** 2

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
    t_values = np.linspace(0,2*np.pi, int(T * 1000))
    phi_values = phi(t_values)


    # 计算z(t')的值
    z_values = [(amplitude * np.pi / 180)*z(phi_prime, t_values) for phi_prime in phi_values]
    # 使用线性插值生成均匀的时间点
    phi_uniform = np.linspace(0, 2*np.pi, int(T * 1000))
    f_interp = interp1d(phi_values, z_values)
    z_uniform = f_interp(phi_uniform)


    # 保存到文件
    with open(os.path.join(folder_name, "control.txt"), "w") as f:
        for value in z_uniform:
            f.write(str(value) + "\n")

    plt.plot(phi_values, z_values, color="green")
    plt.xlabel("φ")
    plt.ylabel("z(φ(t))")
    plt.show()

    return f"Waveform saved to {folder_name}/control.txt"


# Test
generate_waveform(40, 0.8, -0.9, "MMGP_OL7")
