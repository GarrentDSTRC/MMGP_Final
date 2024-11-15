import torch
torch.set_default_tensor_type(torch.FloatTensor)
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from time import time
from scipy.interpolate import griddata
import numpy as np
from scipy.stats import norm
import random
from TWarping import generate_waveform
from pprint import pprint
from gpytorch.priors import NormalPrior
from deap import algorithms, base, creator, tools
from functools import partial
from scipy.spatial.distance import cdist
# Load the new data file without headers
centroids_df = pd.read_csv('Database\centroids.csv', header=None)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
Frame = pd.read_excel('.\ROM\BF_search.xlsx', sheet_name="HL")
Frame2 = pd.read_excel('.\ROM\BF_search.xlsx', sheet_name="HL")
#CT2ETA2_CUT; ETA_CUT
#TestX = torch.FloatTensor(Frame.iloc[:, 0:4].to_numpy()).to(device)
St = torch.linspace(0.6,1.0,5)
ad = torch.linspace(0.1,0.6,6)
phi = torch.linspace(5,40,8)
theta = torch.linspace(0,180,7)
N = torch.linspace(0,9,4)
A = torch.linspace(0,9,4)
CA = torch.linspace(10,35,6)
a,b,c,d,e,f,g=torch.meshgrid(St,ad,phi,theta,N,A,CA)
TestX=torch.as_tensor(list(zip(a.flatten(),b.flatten(),c.flatten(),d.flatten(),e.flatten(),f.flatten(),g.flatten())) )
OLSCALE=1
#UPB=[1.0, 0.6, 40, 180, 1,1,1000]
#LOWB=[0.4, 0.1, 5, -180, -1,-1,100]
UPB=[0.3, 1.3, 85, 180, 0.9,0.9,9,9,35]
LOWB=[0.1, 0.4, 15, -180, -0.9,-0.9,0,0,10]

import time
inittime=time.time()
###################num——task   + single fidelity -multifidelity    ### |task|=multitask

centroids_array = centroids_df.to_numpy()
centroids_tensor = torch.tensor(centroids_array, dtype=torch.float32)
def replace_last_three_with_nearest_class_tensor(matrix):
    """
    将矩阵的每一行最后三个元素替换为最接近的类中心。

    参数:
    matrix (torch.Tensor): 任意维度的输入张量。

    返回:
    torch.Tensor: 修改后的张量。
    """
    # 确保输入是一个 PyTorch 张量
    matrix = torch.tensor(matrix, dtype=torch.float32)
    if len(matrix.shape) == 1:  # 如果输入是一行
        matrix = matrix.unsqueeze(0)  # 添加一个维度变成 (1, n)
    # 获取列数
    n_cols = matrix.shape[1]
    # 遍历每一行
    for row in range(matrix.shape[0]):
        # 处理每一行的最后三个值
        for col in range(n_cols - 3, n_cols):
            # 当前需要替换的值
            value = matrix[row, col]

            # 计算与每个类中心的欧几里得距离
            distances = torch.norm(centroids_tensor - value, dim=1)

            # 找到距离最小的类中心
            nearest_centroid = centroids_tensor[torch.argmin(distances)]

            # 替换值为最近的类中心
            matrix[row, col] = nearest_centroid[col % nearest_centroid.size(0)]

    return matrix

class Normalizer:
    def __init__(self, low_bound, up_bound):
        self.low_bound = torch.tensor(low_bound, dtype=torch.float32).to(device)
        self.up_bound = torch.tensor(up_bound, dtype=torch.float32).to(device)

    def normalize(self, x):
        x=torch.as_tensor(x)
        return (x - self.low_bound) / (self.up_bound - self.low_bound)

    def denormalize(self, norm_x):
        norm_x = torch.as_tensor(norm_x)
        return norm_x * (self.up_bound - self.low_bound) + self.low_bound

"-----------------------FIND POINT-----------------------------------------------"
def findpoint(point,Frame):
    min=99
    minj=0
    for j in range(len(Frame.iloc[:,0])):
        abs=np.sum(np.abs(Frame.iloc[j,0:4].to_numpy()-point))
        if min>abs:
            min=abs
            minj=j
    return Frame.iloc[minj,0:4].to_numpy(),Frame.iloc[minj,4]


normalizer = Normalizer(LOWB, UPB)


def findpointOL(X,num_task=1,mode="experiment"):
#归一化只在这里归一化

    last_col = X[:, -1]  # Extract the last column
    if mode=="experiment" or "CFD" or "experiment_cluster":
        num_p=X.shape[0]
        all_Y=[]
        num_task=np.abs(num_task)
        for i in range(int(num_p/8)):
            for j in range(8):
                generate_waveform(X[i*8+j,0:6].tolist(),r'.\MMGP_OL%d'%(j%8),mode)
                np.savetxt(r'.\MMGP_OL%d\flag.txt'%(j%8), np.array([0]), delimiter=',', fmt='%d')
                fill=np.array([[0,0,0,0,X[i*8+j,6],X[i*8+j,7],X[i*8+j,8],10000 ]])
                np.savetxt(r'.\MMGP_OL%d\dataX.txt' % (j % 8), fill, delimiter=',', fmt='%.2f')
            for j in range(8):
                flag=np.loadtxt(r'.\MMGP_OL%d\flag.txt'%(j%8), delimiter=",", dtype="int")

                while flag==0:
                    try:
                        flag=np.loadtxt(r'.\MMGP_OL%d\flag.txt'%(j%8), delimiter=",", dtype="int")
                    finally:
                        time.sleep(25)
                        print("程序运行时间",(time.time()-inittime)/3600)
                all_Y.append(np.loadtxt(r'.\MMGP_OL%d\dataY.txt'%(j%8), delimiter=",", dtype="float"))
        all_Y=np.asarray(all_Y)
        all_Y[:,1]=all_Y[:,1]*OLSCALE

        if num_task==1:
            return torch.tensor(X).to(device), torch.tensor(all_Y[:,0]).to(device)
        else:
            return torch.tensor(X).to(device), torch.tensor(all_Y).to(device)
    else:
        from pymoo.problems.many.wfg import WFG1
        from pymoo.problems.many.dtlz import DTLZ1
        from pymoo.factory import get_problem
        if mode=="test_WFG" :
            problem = WFG1(n_var=7, n_obj=2)
        else:
            problem = DTLZ1(n_var=7, n_obj=2)
            #problem = get_problem("zdt1", n_var=7)
        if mode=="test_WFG" :
            all_Y = problem.evaluate(np.array(X))  # 计算目标值
        else:
            all_Y = problem.evaluate(np.array(X))  # 计算目标值
            signs = np.sign(all_Y)
            abs_values = np.abs(all_Y)
            all_Y = -np.power(abs_values, 1 / 3) * signs
        return torch.tensor(X).to(device), torch.tensor(all_Y).to(device)
# 定义一个转换后的问题类，继承自原始问题类 #原本函数为正求最小，我全转为负，然后求最大（通过转为正求最小）
from pymoo.core.problem import Problem
class TransformedProblem(Problem):
    # 初始化方法，接收原始问题作为参数，并定义问题的属性
    def __init__(self, problem):
        super().__init__(n_var=problem.n_var,
                         n_obj=problem.n_obj,
                         n_constr=problem.n_constr,
                         xl=problem.xl,
                         xu=problem.xu)
        # 将原始问题保存为类的属性
        self.problem = problem

    # 评估方法，接收决策变量x和输出字典out，并计算目标值和约束值
    def _evaluate(self, x, out, *args, **kwargs):
        # 调用原始问题的评估方法，得到原始目标值和约束值
        self.problem._evaluate(x, out, *args, **kwargs)

        # 将原始目标值取负数，得到转换后的目标值，并赋值给输出字典的"F"键
        out["F"] = -out["F"]
def findpoint_interpolate(point,Frame,num_tasks=1,method="linear"):
    X=[]
    num_tasks=np.abs(num_tasks)
    for i in range(4):
        X.append(Frame.iloc[:,i].to_numpy())
    if num_tasks==1:
        Y=Frame.iloc[:,4].to_numpy()
        return point,griddata(tuple(X), Y, point,method=method)
    else:
        Y = Frame.iloc[:, 4:4+num_tasks].to_numpy()
        value=[]
        for i in range(num_tasks):
            value.append(griddata(tuple(X), Y[:,i], point, method=method))
        return point,np.array(value).T

def infill(model, likelihood, n_points, dict, num_tasks=1, method="error", cofactor=[0.5,0.5], offline=1, device = torch.device("cpu"),y_max=999):
    "num_task<0    ->use the multi-fidelity kernel"
    #num_task=1 Single GP; num_task=0 Raw;num_task=2 Multitask;num_task=-2  and multifidelity ;num_task=-1 multifidelity cofactor 0->EI 1->task
    randomsample=3000
    Result_idx = []
    model.eval()
    likelihood.eval()
    print("num_dict",len(dict))
    selectDict= random.sample(dict, randomsample)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        if num_tasks<0:
            B=TestX[selectDict, :].to(device)
            C=torch.ones( len(TestX[selectDict, :]))
            if num_tasks==-2:
                A=likelihood(*model((B,C),(B,C) ))
                VarS = A[0].variance
                MeanS = A[0].mean
                VarS2=A[1].variance
                MeanS2=A[1].mean
            else:
                A=likelihood(model(B, C))
                VarS = A.variance
                MeanS = A.mean
        else:
            if num_tasks != 1:
                B = TestX[selectDict, :].to(device)
                A = likelihood(*model(B, B))
                VarS = A[0].variance
                MeanS = A[0].mean
                VarS2=A[1].variance
                MeanS2=A[1].mean
            else:
                A = likelihood(model(TestX[selectDict, :]))
                VarS = A.variance
                MeanS = A.mean
        if method == "EI":
            VarS=VarS+1e-05 #prevent var=0
            EI_one = (MeanS-y_max[0]) * torch.FloatTensor(norm.cdf(((MeanS-y_max[0])/VarS).cpu().detach())).to(device)
            EI_two = VarS* torch.FloatTensor(norm.pdf( (MeanS-y_max[0]/VarS).cpu().detach() )).to(device)
            EI = EI_one*cofactor[0] + EI_two*(1-cofactor[0])

            if np.abs(num_tasks)==2:
                VarS2 = VarS2 + 1e-05
                EI_one1 = (MeanS2-y_max[1]) * torch.FloatTensor(norm.cdf(((MeanS2-y_max[1]) / VarS2).cpu().detach())).to(device)
                EI_two1 = VarS2 * torch.FloatTensor(norm.pdf(((MeanS2-y_max[1]) / VarS2).cpu().detach())).to(device)
                EI1 = EI_one1*cofactor[0] + EI_two1*(1-cofactor[0])
                EI=(EI/torch.max(EI))*cofactor[1]+(EI1/torch.max(EI1))*(1-cofactor[1])


            VarS=EI
        if method == "PI":
            VarS=VarS+1e-05 #prevent var=0
            PI = torch.FloatTensor(norm.cdf(((MeanS-y_max[0])/VarS).cpu().detach())).to(device)

            if np.abs(num_tasks)==2:
                VarS2 = VarS2 + 1e-05
                PI1 =  torch.FloatTensor(norm.cdf(((MeanS2-y_max[1]) / VarS2).cpu().detach())).to(device)
                PI=(PI/torch.max(PI))*cofactor[1]+(PI1/torch.max(PI1))*(1-cofactor[1])
            VarS=PI
        if method == "UCB":
            k=1.68
            UCB=k*VarS+MeanS
            if np.abs(num_tasks) == 2:
                UCB1 = k * VarS2 + MeanS2
                UCB= (UCB / torch.max(UCB)) * cofactor[1] + (UCB1 / torch.max(UCB1)) * (1 - cofactor[1])
            VarS=UCB

        for i in range(n_points):
            Result_idx.append(selectDict[torch.argmax(VarS).item()])
            VarS[torch.argmax(VarS).item()] = -999
            print("remove", Result_idx[i])
        for i in range(n_points):
            dict.remove(Result_idx[i])
        if offline==1:
            return torch.tensor(Frame.iloc[Result_idx, 0:4].to_numpy()).to(device), torch.tensor(
            Frame.iloc[Result_idx, 4:4 + np.abs(num_tasks)].to_numpy()).to(device)
        else:
            X =TestX[Result_idx, :]
            X,Y=findpointOL(X,num_task=num_tasks)
            return X,Y

def infillGA(model, likelihood, n_points, dict, num_tasks=1, method="error", cofactor=[0.5,0.5], offline=1, device = torch.device("cpu"),y_max=999,train_x=[],testmode="experiment",final_population_X=[],norm=None):
    "num_task<0    ->use the multi-fidelity kernel"
    #num_task=1 Single GP; num_task=0 Raw;num_task=2 Multitask;num_task=-2  and multifidelity ;num_task=-1 multifidelity
    # Create a new Individual class
    creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Convert final population to list of Individuals
    final_population_individuals = [creator.Individual(x) for x in train_x]



    # Evaluate each individual in the population
    for i,individual in enumerate(final_population_individuals):
        # If testmode is "experiment", classify and replace the last three elements
        if testmode == "experiment_cluster":
            clustered = replace_last_three_with_nearest_class_tensor(individual)
            individual[:] = clustered.values
        # Call your evaluateEI function here, replace y_max and cofactor with the actual values
        individual.fitness.values = evaluateEI(individual,
                                               model=model,
                                               likelihood=likelihood,
                                               y_max=y_max,
                                               cofactor=cofactor,
                                               num_task=num_tasks,
                                               )
    popsize = 600
    cxProb = 0.7
    mutateProb = 0.2
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attribute, n=100)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluateEI, model=model, likelihood=likelihood, y_max=y_max, cofactor=cofactor,num_task=num_tasks)
    toolbox.decorate('evaluate', tools.DeltaPenalty(feasibleMT, -1e3))  # death penalty
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # 运行遗传算法
    pop = final_population_individuals
    hof = tools.HallOfFame(32)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, stats=stats, halloffame=hof,
                                       verbose=True)
    # algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=100, cxpb=0.8, mutpb=1.0/NDIM, ngen=100)
    # 计算Pareto前沿集合
    for i in range(1, 13):
        fronts = tools.emo.sortLogNondominated(pop, popsize, first_front_only=False)
        # pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof,
        # verbose=True)
        for front in fronts:
            tools.emo.assignCrowdingDist(front)
        pop = []
        for front in fronts:
            pop += front

        # 复制
        pop = toolbox.clone(pop)
        # 基于拥挤度的选择函数用来实现精英保存策略
        pop = tools.selNSGA2(pop, k=popsize, nd='standard')
        # 创建子代
        offspring = toolbox.select(pop, popsize)
        offspring = toolbox.clone(offspring)
        offspring = algorithms.varAnd(offspring, toolbox, cxProb, mutateProb)

        # Evaluate the population
        if testmode == "experiment_cluster":
            for individual in offspring:
                clustered = replace_last_three_with_nearest_class_tensor(individual)
                individual[:] = clustered.values

        # 记录数据-将stats的注册功能应用于pop，并作为字典返回
        record = stats.compile(pop)
        logbook.record(gen=i, **record)

        # 合并父代与子代
        pop = pop + offspring
        # 评价族群-更新新族群的适应度
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        print(logbook.stream)
################################################

    pareto_front_ALL = tools.emo.sortLogNondominated(pop, len(pop), first_front_only=False)


    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        plt.clf()
        exploitation = []
        exploration = []
        for ind in pareto_front_ALL[0]:
            exploitation.append(ind.fitness.values[0])
            exploration.append(ind.fitness.values[1])
            plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'r.', ms=2)
        plt.xlabel('exploitaion', fontsize=14)
        plt.ylabel('exploration', fontsize=14)
        plt.tight_layout()
        plt.savefig("taskRelaiton.png", dpi=300)
        df = pd.DataFrame({'exploitation': exploitation, 'exploration': exploration})
        df.to_csv('taskRelaiton.csv', index=False)

        candidates = []
        for pareto_front in pareto_front_ALL:
            sorted_front = sorted(pareto_front, key=lambda ind: ind.fitness.values[0] + ind.fitness.values[1],reverse=True)
            for ind in sorted_front:
                #candidate = [round(x, 4) * (UPB[i] - LOWB[i])  + LOWB[i] for i, x in enumerate(ind)]
                candidate = [round(x.item(), 2)  for i, x in enumerate(ind)]
                if candidate not in candidates and candidate not in np.round(train_x.tolist(),2).tolist() :
                    candidates.append(candidate)
                if len(candidates) == n_points:
                    break
                if len(candidates) >= n_points:
                    candidates=candidates[0:n_points]
                    break
        if len(candidates) < n_points:
            candidates = candidates[0:len(candidates)-len(candidates)%8]

        X = torch.tensor(candidates).to(device).to(torch.float32)
        denorm_X=norm.denormalize(X)
        print("PLAN TO SEARCH",denorm_X)
        POINT,Y=findpointOL(denorm_X,num_task=num_tasks,mode=testmode)
        return X,Y

def UpdateCofactor(model,likelihood,X,Y,cofactor,maxmin,MFkernel=0):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if MFkernel==0:
            A = likelihood(*model(X, X,X))
        else:
            I=torch.ones(X.shape[0]).to(torch.float32)
            A =likelihood(*model((X, I), (X, I), (X, I)))
        M = torch.mean(torch.abs(Y - torch.cat([A[0].mean.unsqueeze(1), A[1].mean.unsqueeze(1),A[2].mean.unsqueeze(1)], dim=1)), dim=0)
        cofactor[1][0] = M[0] / maxmin[0] / (M[0] / maxmin[0] + M[1] / maxmin[1]+M[2] / maxmin[2])
        cofactor[1][1] = M[1] / maxmin[1] / (M[0] / maxmin[0] + M[1] / maxmin[1]+M[2] / maxmin[2])
        cofactor[1][2] = M[2] / maxmin[2] / (M[0] / maxmin[0] + M[1] / maxmin[1] + M[2] / maxmin[2])
        f = open("./cofactor.txt", "a", encoding="utf - 8")
        f.writelines((str(cofactor[1]) + ",", str(M[0].item()) + ",", str(M[1].item()) + "\n"))
        f.close()

        if MFkernel == 0:
            return cofactor
        else: return cofactor,M

#________________________________________GA
def evaluateMT(individual,model,likelihood):
    model.eval()
    likelihood.eval()
    ind=[0]*len(UPB)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(len(UPB)):
            ind[i]=individual[i]*(UPB[i]-LOWB[i])+LOWB[i]
        ind=torch.tensor(ind).to(device).to(torch.float32).unsqueeze(0)
        A=likelihood(*model( ind,  ind))
    return   A[0].mean.item(),A[1].mean.item()
def feasibleMT(ind):
    # 判定解是否满足约束条件
    # 如果满足约束条件，返回True，否则返回False
    for i in range(len(UPB)) :
        if   (1-ind[i])<=0:
            return False
        if   (ind[i]-0)<=0:
            return False
    return  True
def evaluateEI(individual,model,likelihood,y_max,cofactor,num_task=2):
    model.eval()
    likelihood.eval()
    ind=[0]*len(UPB)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(len(UPB)):
            #ind[i]=individual[i]*(UPB[i]-LOWB[i])+LOWB[i]
            ind[i]=individual[i]
        ind=torch.tensor(ind).to(device).to(torch.float32).unsqueeze(0)
        with gpytorch.settings.cholesky_jitter(1e-0):
            if np.abs(num_task)==2:
                if num_task==-2:
                    test_i_task2 = torch.full((ind.shape[0], 1), dtype=torch.long, fill_value=1)
                    A = likelihood(*model((ind,test_i_task2),(ind,test_i_task2)))
                else:
                    A=likelihood(*model( ind,  ind))
            else:
                if num_task==-3:
                    test_i_task2 = torch.full((ind.shape[0], 1), dtype=torch.long, fill_value=1)
                    A = likelihood(*model((ind,test_i_task2),(ind,test_i_task2),(ind,test_i_task2)))
                else:
                    A=likelihood(*model( ind,  ind, ind))
                # 为 A[2] 计算 EI
                VarS3 = A[2].variance
                MeanS3 = A[2].mean
                VarS3 = VarS3 + 1e-05
                EI_one2 = (MeanS3 - y_max[2]) * torch.FloatTensor(norm.cdf(((MeanS3 - y_max[2]) / VarS3).cpu().detach())).to(device)
                EI_two2 = VarS3 * torch.FloatTensor(norm.pdf(((MeanS3 - y_max[2]) / VarS3).cpu().detach())).to(device)

        VarS = A[0].variance
        MeanS = A[0].mean
        VarS = VarS + 1e-05  # prevent var=0
        EI_one = (MeanS - y_max[0]) * torch.FloatTensor(norm.cdf(((MeanS - y_max[0]) / VarS).cpu().detach())).to(device)
        EI_two = VarS * torch.FloatTensor(norm.pdf((MeanS - y_max[0] / VarS).cpu().detach())).to(device)
        #ct

        VarS2 = A[1].variance
        MeanS2 = A[1].mean
        VarS2 = VarS2 + 1e-05
        EI_one1 = (MeanS2 - y_max[1]) * torch.FloatTensor(norm.cdf(((MeanS2 - y_max[1]) / VarS2).cpu().detach())).to(
            device)
        EI_two1 = VarS2 * torch.FloatTensor(norm.pdf(((MeanS2 - y_max[1]) / VarS2).cpu().detach())).to(device)
        #eta
    if np.abs(num_task)==2:
        return  ( EI_one*cofactor[1]+EI_one1*(1-cofactor[1])).item(),(EI_two*cofactor[1]+EI_two1*(1-cofactor[1])).item()
    elif np.abs(num_task)==3:
        return  ( EI_one*cofactor[1][0]+EI_one1*cofactor[1][1]+EI_one2*cofactor[1][2]).item(),(EI_two*cofactor[1][0]+EI_two1*cofactor[1][1]+EI_two2*cofactor[1][2]).item()



def evaluateEISO(individual,model,likelihood,y_max,cofactor):
    model.eval()
    likelihood.eval()
    ind=[0]*len(UPB)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(len(UPB)):
            ind[i]=individual[i]*(UPB[i]-LOWB[i])/(UPB[i]-1)+LOWB[i]
        ind=torch.tensor(ind).to(device).to(torch.float32).unsqueeze(0)
        A=likelihood(*model( ind,  ind))

        VarS = A[0].variance
        MeanS = A[0].mean
        VarS2 = A[1].variance
        MeanS2 = A[1].mean

        VarS = VarS + 1e-05  # prevent var=0
        EI_one = (MeanS - y_max[0]) * torch.FloatTensor(norm.cdf(((MeanS - y_max[0]) / VarS).cpu().detach())).to(device)
        EI_two = VarS * torch.FloatTensor(norm.pdf((MeanS - y_max[0] / VarS).cpu().detach())).to(device)
        EI = EI_one * cofactor[0] + EI_two * (1 - cofactor[0])

        VarS2 = VarS2 + 1e-05
        EI_one1 = (MeanS2 - y_max[1]) * torch.FloatTensor(norm.cdf(((MeanS2 - y_max[1]) / VarS2).cpu().detach())).to(
            device)
        EI_two1 = VarS2 * torch.FloatTensor(norm.pdf(((MeanS2 - y_max[1]) / VarS2).cpu().detach())).to(device)
        EI1 = EI_one1 * cofactor[0] + EI_two1 * (1 - cofactor[0])

    return   EI.item(),EI1.item()
if __name__=="__main__":
    pass