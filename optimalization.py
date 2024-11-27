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
training_iterations = 30#33#5
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

# Here we have two iterms that we're passing in as train_inputs
likelihood1 = gpytorch.likelihoods.GaussianLikelihood().to(device)
#50: 0:2565
model1 = MultiFidelityGPModel((full_train_x[:,:], full_train_i[:,:]), full_train_y[:,0], likelihood1).to(device)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood().to(device)
model2 = MultiFidelityGPModel((full_train_x[:,:], full_train_i[:,:]), full_train_y[:,1], likelihood2).to(device)
likelihood3 = gpytorch.likelihoods.GaussianLikelihood().to(device)
model3 = MultiFidelityGPModel((full_train_x[:,:], full_train_i[:,:]), full_train_y[:,2], likelihood3).to(device)

model = gpytorch.models.IndependentModelList(model1, model2,model3).to(device)
likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood,model3.likelihood)

print(model)


# "Loss" for GPs - the marginal log likelihood
from gpytorch.mlls import SumMarginalLogLikelihood
mll = SumMarginalLogLikelihood(likelihood, model)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters


cofactor = [0.5, [0.3, 0.3, 0.3]]
for i in range(Episode):
    print(  "Episode%d-point %d : %d "%(i, torch.sum(full_train_i).item(),len(full_train_i)-torch.sum(full_train_i).item())   )
    if os.path.exists(path5):
        state_dict = torch.load(path5)
        model.load_state_dict(state_dict)
    else:
        pass
    # for j in range(training_iterations):
    #     optimizer.zero_grad()
    #     output = model(*model.train_inputs)
    #     loss = -mll(output, model.train_targets)
    #     loss.backward(retain_graph=True)
    #     print('Iter %d/%d - Loss: %.3f' % (j + 1,training_iterations, loss.item()))
    #     optimizer.step()
    # torch.save(model.state_dict(), path5)
model.eval()
likelihood.eval()



# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.population import Population


# Define a custom problem class that takes the observed predictions as objectives
class MyProblem(Problem):

    def __init__(self, ):
        super().__init__(n_var=UPBound.shape[0],
                         n_obj=3,
                         n_constr=0,
                         xu=UPBound.shape[0]*[1],
                         xl=UPBound.shape[0]*[0])


    def _evaluate(self, x, out, *args, **kwargs):
        # Use the observed predictions as objectives
        test_x = torch.tensor(x).to(torch.float32)
        test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.float32, fill_value=1)

        # Make predictions - one task at a time
        # We control the task we cae about using the indices

        # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
        # See https://arxiv.org/abs/1803.06058
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_yH = likelihood(*model((test_x, test_i_task2), (test_x, test_i_task2), (test_x, test_i_task2)))
            observed_pred_yHC = -1*np.array(observed_pred_yH[0].mean.tolist())  # ct high
            observed_pred_yHL = -1*np.array(observed_pred_yH[1].mean.tolist())  # CL high
            observed_pred_yHE = -1*np.array(observed_pred_yH[2].mean.tolist())  # eta high

        N=np.array([observed_pred_yHC,observed_pred_yHL,observed_pred_yHE]).T
        #print(N)
        out["F"] =N
        #N1=-N[:, 1]-0.97
        #out["G"] = [N[:, 0]]         #.reshape(-1, 3, 1)
        #out["G"] = N1.reshape(-1, 1)
# # Create an instance of the problem with the observed predictions
problem = MyProblem()
#
# # Define the reference directions for the Pareto front
from pymoo.util.ref_dirs import get_reference_directions
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# Initialize the algorithm with the last 150 samples from full_train_x
pop = Population.new("X", full_train_x[-200:,:].numpy())
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
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_yH = likelihood(*model((x, full_train_i[-6:,:]), (x, full_train_i[-6:,:]), (x, full_train_i[-6:,:])))
    observed_pred_yHC =  torch.atanh(observed_pred_yH[0].mean/GPscale) *GPscale2 # ct high
    observed_pred_yHL =  torch.atanh(observed_pred_yH[1].mean/GPscale) *GPscale2 # CL high
    observed_pred_yHE =  torch.atanh(observed_pred_yH[2].mean/GPscale) *GPscale2 # eta high
combined_matrix = torch.stack([observed_pred_yHC, observed_pred_yHL, observed_pred_yHE]).T
print("combined_matrix",combined_matrix)

algorithm = NSGA2(pop_size=900, eliminate_duplicates=True,sampling=pop)

# Minimize the problem using the algorithm and the initial population
res = minimize(problem,
               algorithm,
               ("n_gen", 13),#70
               seed=1,
)
res_F_tensor = torch.tensor(res.F, dtype=torch.float32)
res_F_tensor = torch.atanh(res_F_tensor/GPscale)*GPscale2
res.F = res_F_tensor.numpy()  # 转换回 numpy 数组

# Get the last population from the result object
last_pop = res.pop

# Save the last population to the path
np.save(pathpop, last_pop)



# 假设 res.F 是一个已经存在的数组，包含了你想要绘制的坐标数据
# res.F[:, 0], res.F[:, 1], res.F[:, 2] 分别对应 x, y, z 轴的数据

fig = plt.figure()  # 创建一个图形对象
ax = fig.add_subplot(111, projection='3d')  # 添加一个三维子图

# 绘制三维散点图
ax.scatter(-1*res.F[:, 0], -1*res.F[:, 1], -1*res.F[:, 2], c="blue", marker="o", s=20)

# 设置坐标轴标签
ax.set_xlabel('ct', fontsize=22, fontfamily='Times New Roman')
ax.set_ylabel('cl', fontsize=22, fontfamily='Times New Roman')
ax.set_zlabel('η', fontsize=22, fontfamily='Times New Roman')  # 假设第三个维度是 η

# 设置坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=14)

# 设置全局字体
plt.rc('font', family='Times New Roman')

# 调整子图位置，以便显示标签
plt.subplots_adjust(left=0.15, bottom=0.15)

# 保存图形为高分辨率的 PDF 文件
plt.savefig("pareto_front_3d.pdf", dpi=300)




# Import the pandas library for data manipulation
import pandas as pd

# Get the Pareto front solutions from the result object
pf = res.opt

# Get the decision variables and objective values of the Pareto front solutions
X = pf.get("X")
F = np.arctan(pf.get("F")/-GPscale)*GPscale2

# Create a pandas dataframe with the decision variables and objective values
df = pd.DataFrame(np.hstack([X, F]), columns=["st",	"ad","theta","phi","alpha1","alpha2","m","p","t", "ct", "cl","eta"])

# Save the dataframe to a csv file
df.to_csv("pareto_front.csv", index=False)