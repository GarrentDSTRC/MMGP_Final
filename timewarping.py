import numpy as np
import matplotlib.pyplot as plt

# 参数定义
alpha = 1  # 这里取alpha为1，您可以根据需要进行调整

# y(t)定义
def y(t):
    return np.sin(t)

# φ(t')定义
def phi(t):
    return t + alpha * np.sin(t)**2

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
t_values = np.linspace(0, 2*np.pi, 1000)
y_values = y(t_values)
phi_values = phi(t_values)

# 计算z(t')的值
z_values = [z(phi_prime, t_values) for phi_prime in phi_values]

# 绘制y(t), φ(t)和z(t')
plt.figure(figsize=(6, 6))

plt.subplot(3, 1, 1)
plt.plot(t_values, y_values)
plt.xlabel("t")
plt.ylabel("y(t)")

plt.subplot(3, 1, 2)
plt.plot(t_values, phi_values, color="orange")
plt.xlabel("t")
plt.ylabel("φ")

plt.subplot(3, 1, 3)
plt.plot(phi_values, z_values, color="green")
plt.xlabel("φ")
plt.ylabel("z(φ)")

plt.tight_layout()
plt.show()
