import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# 参数定义
alpha = 0.5 # 这里取alpha为1，您可以根据需要进行调整

# y(t)定义
def y(t):
    return np.sin(t)

# φ(t)输入t得到φ
def phi(t): 
    return t + alpha * np.sin(t)**2
    #return t - alpha * np.sin(0.5*t)**2
    # return t - alpha * np.sin(0.5 * (t+np.pi/2)) ** 2
# 使用数值方法找到φ^-1(t')的对应t值       输入φ得到t
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
t_values = np.linspace(0, 2*np.pi, 1000)
y_values = y(t_values)
phi_values = phi(t_values)

# 计算z(t')的值
z_values = [z(phi_prime, t_values) for phi_prime in phi_values]
# 使用线性插值生成均匀的时间点
phi_uniform = t_values
f_interp = interp1d(phi_values, z_values, fill_value='extrapolate')
z_uniform = f_interp(phi_uniform)

# 绘制y(t), φ(t)和z(t')
plt.figure(figsize=(4, 6))

plt.subplot(3, 1, 1)
plt.plot(t_values, y_values)
plt.xlabel("t")
plt.ylabel("y(t)")

plt.subplot(3, 1, 2)
plt.plot(t_values, phi_values, color="orange")
plt.xlabel(r"t")
plt.ylabel(r"$\phi_{1}^'$")

plt.subplot(3, 1, 3)
plt.plot(phi_uniform, z_uniform, color="green")
plt.xlabel("t'")
plt.ylabel(r"$z_{1}(t')$")

plt.tight_layout()
plt.show()
