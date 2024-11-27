#全局优化
import math
import numpy as np
import torch
import gpytorch
from pyKriging.samplingplan import samplingplan
import time
import os
from GPy import *
from Opt import *
from Kernels import *
from gpytorch.lazy import InterpolatedLazyTensor
from mpl_toolkits.mplot3d import Axes3D

from gpytorch import kernels, means, models, mlls, settings, likelihoods, constraints, priors
from gpytorch import distributions as distr
path5='.\Database\multifidelity_database.pth'
pathpop=r".\Database\pop.npy"
path1csv=r".\Database\saveX_gpytorch_multifidelity_multitask.csv"
path2csv=r".\Database\saveI_gpytorch_multifidelity_multitask.csv"
path3csv=r".\Database\saveY_gpytorch_multifidelity_multitask.csv"
path4csv=r".\Database\saveTestXdict_gpytorch_multifidelity_multitask.csv"
#lowfidelity
#pathx2=r".\ROM\E3\saveX_gpytorch_multi_EI_MS 928.npy"
#pathy2=r".\ROM\E3\savey_gpytorch_multi_EI_MS 928.npy"
pathx2=r".\Database\saveX_gpytorch_multi_EI_MS.npy"
pathy2=r".\Database\savey_gpytorch_multi_EI_MS.npy"

path1H=r".\Database\saveX_gpytorch_multi_EI_MS_H.npy"
path2H=r".\Database\savey_gpytorch_multi_EI_MS_H.npy"

# We make an nxn grid of training points spaced every 1/(n-1) on [0,1]x[0,1]
# n = 250
init_sample = 8*15
UpB=len(Frame.iloc[:, 0].to_numpy())-1
LowB=0
Infillpoints=8*2
training_iterations = 290#33#5
num_tasks=-3
num_input=len(UPB)
Episode=1
LowSample=1800
testsample=140
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device( "cpu")
Offline=0
testmode="CFD"
UPBound = np.array(UPB).T
LOWBound = np.array(LOWB).T
dict = [i for i in range(TestX.shape[0])]
GPscale=3
GPscale2=3
if os.path.exists(path1csv):
    full_train_x=torch.FloatTensor(np.loadtxt(path1csv, delimiter=','))
    full_train_i=torch.FloatTensor(np.loadtxt(path2csv,delimiter=','))
    full_train_y=torch.FloatTensor(np.loadtxt(path3csv, delimiter=','))
    full_train_i=torch.unsqueeze(full_train_i,dim=1)
    full_train_y =GPscale* torch.tanh(full_train_y/GPscale2)
    if os.path.exists(path4csv):
        dict = np.loadtxt(path4csv,  delimiter=',').astype(int).tolist()

# 定义神经网络
model_nn = LargeFeatureExtractor(dim1=UPBound.shape[0], dim2=3)  # 3是输出维度 (3个任务)

# 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)


# 定义保存和加载模型的路径
model_save_path = './model_nn.pth'

# 训练神经网络
model_nn.train()
for i in range(training_iterations):
    optimizer.zero_grad()

    # 前向传播
    output = model_nn(full_train_x)

    # 计算损失
    loss = loss_fn(output, full_train_y)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 打印损失
    print(f'Iteration {i + 1}/{training_iterations}, Loss: {loss.item()}')

    # 每隔一定的迭代保存一次模型参数
    if (i + 1) % 5 == 0:  # 例如每5次迭代保存一次
        torch.save(model_nn.state_dict(), model_save_path)
        print(f'Model parameters saved at iteration {i + 1}')

# 训练完成后，保存最终的模型参数
torch.save(model_nn.state_dict(), model_save_path)
print(f'Final model parameters saved to {model_save_path}')

#pop.set("F", problem.evaluate(pop.get("X")))
# Create an instance of the NSGA-II algorithm

x=[[0.829999983,0.419999987,0.899999976,0.970000029,0.589999974,0.540000021,0.699999988,0.379999995,0.899999976],
[0.860000014,0.99000001,0.529999971,0.430000007,0.899999976,0.99000001,0.699999988,0.379999995,0.899999976],
[0.860000014,0.920000017,0.079999998,0.99000001,0.899999976,0.99000001,0.699999988,0.379999995,0.899999976],
[0.829999983,0.419999987,0.899999976,0.970000029,0.5,0.5,0.699999988,0.379999995,0.899999976],
[0.860000014,0.99000001,0.529999971,0.430000007,0.5,0.5,0.699999988,0.379999995,0.899999976],
[0.860000014,0.920000017,0.079999998,0.99000001,0.5,0.5,0.699999988,0.379999995,0.899999976],
   ]
x=torch.tensor(x)
model_nn.eval()
with torch.no_grad():
    # 对新数据进行预测
    observed_pred = model_nn(x)
    observed_pred_yHC = observed_pred[:, 0]  # 第一个任务
    observed_pred_yHL = observed_pred[:, 1]  # 第二个任务
    observed_pred_yHE = observed_pred[:, 2]  # 第三个任务
combined_matrix = torch.atanh(torch.stack([observed_pred_yHC, observed_pred_yHL, observed_pred_yHE]).T/GPscale) *GPscale2
print("combined_matrix",combined_matrix)

