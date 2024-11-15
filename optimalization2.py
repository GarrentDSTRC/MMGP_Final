#双层GA
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
training_iterations = 13#33#5
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

if os.path.exists(path1csv):
    full_train_x=torch.FloatTensor(np.loadtxt(path1csv, delimiter=','))
    full_train_i=torch.FloatTensor(np.loadtxt(path2csv,delimiter=','))
    full_train_y=torch.FloatTensor(np.loadtxt(path3csv, delimiter=','))
    full_train_i=torch.unsqueeze(full_train_i,dim=1)
    full_train_y = torch.tanh(full_train_y)

    last_pop=np.load(pathpop, allow_pickle=True)
    # Get the F and X values from the population
    F = np.array([ind.get("F") for ind in last_pop])
    X = np.array([ind.get("X") for ind in last_pop])
    # Sum the two dimensions of F
    F_sum = F.sum(axis=1)
    # Sort the F_sum and X by ascending order of F_sum
    sorted_indices = np.argsort(F_sum)
    F_sum_sorted = F_sum[sorted_indices]
    X_sorted = X[sorted_indices]
    # Get the X numpy array
    pop = torch.FloatTensor(X_sorted)

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
        for j in range(training_iterations):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward(retain_graph=True)
            print('Iter %d/%d - Loss: %.3f' % (j + 1,training_iterations, loss.item()))
            optimizer.step()
        torch.save(model.state_dict(), path5)
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

class InnerProblem(Problem):
    def __init__(self, outer_variables):
        super().__init__(n_var=6, n_obj=3, n_constr=0, xu=6*[1], xl=6*[0])
        self.outer_variables = outer_variables
        self.csv_file_path ="all_PREDICT.csv"

    def _evaluate(self, x, out, *args, **kwargs):
        # Join the inner and outer variables
        full_variables = np.concatenate([x, np.repeat(self.outer_variables[None, :], len(x), axis=0)], axis=1)

        # Use these variables as inputs to your GP model
        test_x = torch.tensor(full_variables).to(torch.float32)
        test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_yH = likelihood(*model((test_x, test_i_task2), (test_x, test_i_task2),(test_x, test_i_task2)))
            observed_pred_yHC = -np.array(observed_pred_yH[0].mean.tolist())  # ct high
            observed_pred_yHL = - np.array(observed_pred_yH[1].mean.tolist()) # CL high
            observed_pred_yHE = - np.array(observed_pred_yH[2].mean.tolist())  # eta high
        N=np.array([observed_pred_yHC,observed_pred_yHL,observed_pred_yHE]).T
        results = np.hstack((test_x.numpy(), N))
        df = pd.DataFrame(results)
        # Incrementally write to the CSV file
        df.to_csv(self.csv_file_path, mode='a', header=not pd.io.common.file_exists(self.csv_file_path), index=False)
        out["F"] =N



class OuterProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=3, n_constr=0, xu=np.array([0,0,0]), xl=np.array([0,0,0]))
#,type_var=np.int
    def _evaluate(self, x, out, *args, **kwargs):
        results = []
        i=0
        #x = np.round(x).astype(int)  # Convert the variables to integers
        for variables in x:
            i=i+1
            problem = InnerProblem(variables)
            pop_2 = Population.new("X", pop[:, :6].numpy())
            algorithm = NSGA2( pop_size=1000, eliminate_duplicates=True,sampling=pop_2)
            res = minimize(problem, algorithm, termination=("n_gen", 15),seed=1)
            max_values = np.min(res.F, axis=0)  # Get the maximum value for each objective
            print(i,max_values)
            results.append(max_values)
        out["F"] = np.array(results)
from pymoo.core.callback import Callback

from mpl_toolkits.mplot3d import Axes3D


class SaveHistoryAndPlot3D(Callback):
    def __init__(self, plot_intervals):
        super().__init__()
        self.history = []
        self.plot_intervals = plot_intervals

    def notify(self, algorithm):
        self.history.append(algorithm.pop)
        n_gen = algorithm.n_gen  # Current generation
        if n_gen in self.plot_intervals:
            self.plot_current_population(algorithm.pop, n_gen)

    def plot_current_population(self,pop, n_gen):
        F = np.array([ind.get("F") for ind in pop])
        X = np.array([ind.get("X") for ind in pop])

        # Plot for CT (First Objective)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=F[:, 0])
        plt.colorbar(sc, ax=ax, label="CT")
        ax.set_title(f"Generation {n_gen} (CT)")
        plt.savefig(f"plot_gen_{n_gen}_CT_3D.png")
        plt.close()

        # Save data for CT plot
        df_ct = pd.DataFrame({'X_0': X[:, 0], 'X_1': X[:, 1], 'X_2': X[:, 2], 'CT': F[:, 0]})
        df_ct.to_csv(f"data_gen_{n_gen}_CT.csv", index=False)

        # Plot for CL (Second Objective)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=F[:, 1])
        plt.colorbar(sc, ax=ax, label="CL")
        ax.set_title(f"Generation {n_gen} (CL)")
        plt.savefig(f"plot_gen_{n_gen}_CL_3D.png")
        plt.close()

        # Save data for CL plot
        df_cl = pd.DataFrame({'X_0': X[:, 0], 'X_1': X[:, 1], 'X_2': X[:, 2], 'CL': F[:, 1]})
        df_cl.to_csv(f"data_gen_{n_gen}_CL.csv", index=False)

        # Plot for Eta (Second Objective)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=F[:, 2])
        plt.colorbar(sc, ax=ax, label="Eta")
        ax.set_title(f"Generation {n_gen} (Eta)")
        plt.savefig(f"plot_gen_{n_gen}_eta_3D.png")
        plt.close()

        # Save data for Eta plot
        df_cl = pd.DataFrame({'X_0': X[:, 0], 'X_1': X[:, 1], 'X_2': X[:, 2], 'Eta': F[:, 2]})
        df_cl.to_csv(f"data_gen_{n_gen}_Eta.csv", index=False)

# Choose generations at which to save plots
plot_intervals = [1, 2,3,4,5,6,7,8,9, 10]  # Beginning, middle, and end

# Initialize callback
callback = SaveHistoryAndPlot3D(plot_intervals)




# Run the outer GA
problem = OuterProblem()
pop_1 = Population.new("X", np.concatenate((pop[0:3,6:].numpy(), pop[15:80, 6:].numpy())))
algorithm = NSGA2(pop_size=50, eliminate_duplicates=True,sampling=pop_1)
# Run optimization with callback
res = minimize(problem, algorithm, termination=("n_gen", 10), callback=callback, seed=1)

res_F_tensor = torch.tensor(res.F, dtype=torch.float32)
res_F_tensor = torch.atanh(res_F_tensor)
res.F = res_F_tensor.numpy()  # 转换回 numpy 数组

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot with the third dimension added
ax.scatter(-1*res.F[:, 0], -1*res.F[:, 1], -1*res.F[:, 2], c="blue", marker="o", s=20)
# Setting labels and title
ax.set_xlabel("f1", fontsize=22, fontfamily='Times New Roman')
ax.set_ylabel("f2", fontsize=22, rotation=0, fontfamily='Times New Roman')
ax.set_zlabel("f3", fontsize=22, fontfamily='Times New Roman')
# ax.set_title("3D Pareto Front", fontsize=18, fontfamily='Times New Roman')
# Setting font properties for all text
plt.rc('font', family='Times New Roman')
# Setting tick label size
ax.tick_params(axis='both', which='major', labelsize=14)
# Adjusting the plot
plt.subplots_adjust(bottom=0.2)
# Save the plot as a high-resolution PDF file
plt.savefig("3D_pareto_front.pdf", dpi=300)

# Show the plot on screen
#plt.show()


# Import the pandas library for data manipulation
import pandas as pd

# Get the Pareto front solutions from the result object
pf = res.opt

# Get the decision variables and objective values of the Pareto front solutions
X = pf.get("X")
F = -pf.get("F")
print(X,F)
# Create a pandas dataframe with the decision variables and objective values
df = pd.DataFrame(np.hstack([X, F]), columns=["m","p","t", "ct", "cl","ETA"])

# Save the dataframe to a csv file
df.to_csv("pareto_frontDOUBLE.csv", index=False)

last_pop = res.pop
F = np.array([ind.get("F") for ind in last_pop])
X = np.array([ind.get("X") for ind in last_pop])
df = pd.DataFrame(np.hstack([X, F]), columns=["m","p","t", "ct", "cl","ETA"])
df.to_csv("popDOUBLE.csv", index=False)