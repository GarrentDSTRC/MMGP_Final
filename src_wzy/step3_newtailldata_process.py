import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取合并后的 CSV 文件
merged_df = pd.read_csv("merged_all_data.csv")

# 过滤出 in_index = 60 的数据
# merged_df = merged_df[merged_df["in_index"] == 60]

# 进行异常值处理（使用 IQR 方法）
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 7 * IQR
    upper_bound = Q3 + 7 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 处理 ct, cl, eff 三个列的异常值
for col in ["ct", "cl", "eff"]:
    merged_df = remove_outliers(merged_df, col)

# 筛选符合条件的数据点
filtered_df = merged_df[(merged_df["ct"] > 0) & (merged_df["eff"] > 0) & (merged_df["eff"] < 1)]

# 按照不同的 eff 范围进行随机抽样
transition1 = filtered_df[(filtered_df["eff"] > 0.05) & (filtered_df["eff"] < 0.085)].sample(frac=0.7, random_state=42)
transition2 = filtered_df[(filtered_df["eff"] > 0.025) & (filtered_df["eff"] < 0.05)].sample(frac=0.4, random_state=42)
transition3 = filtered_df[(filtered_df["eff"] > 0.0) & (filtered_df["eff"] < 0.025)].sample(frac=0.15, random_state=42)


BC_data_eff = merged_df[(merged_df["ct"] > 0) & (merged_df["ct"] < 0.08) & (merged_df["eff"] > 0.052) & (merged_df["eff"] < 0.18)]
# BC_data_fx = merged_df[(merged_df["ct"] > 0.08) & (merged_df["eff"] > 0)  & (merged_df["eff"] < 0.18)]

BC_data_fy = merged_df[(merged_df["ct"] > 0.001) & (merged_df["cl"] > 0.1)  & (merged_df["cl"] < 1.0)]
# BC_data_fy2 = merged_df[(merged_df["ct"] > 0.09) & (merged_df["cl"] > 0.01)  & (merged_df["cl"] < 1.0)]

# expert_data1= merged_df[(merged_df["ct"] > 0) & (merged_df["ct"] < 0.08) & (merged_df["eff"] > 0.038) & (merged_df["eff"] < 0.18)]
# expert_data2= merged_df[(merged_df["ct"] > 0.08) & (merged_df["eff"] < 0.18)]
# expert_data1= merged_df[(merged_df["ct"] > 0) & (merged_df["ct"] < 0.08) & (merged_df["eff"] > 0.038) & (merged_df["eff"] < 0.18)]
expert_data1 = merged_df[(merged_df["ct"] > -0.01) & (merged_df["cl"] > 0.08)]
expert_data2= merged_df[(merged_df["ct"] > 0.086) & (merged_df["cl"] > 0.01)]


# 合并数据集
transition_model_df1 = pd.concat([transition1, transition2, transition3])
expert_model_df1 = pd.concat([expert_data1, expert_data2])


# 仅保留 index 和 in_index 列
transition_model = transition_model_df1[["index", "in_index", "sensor_data_num", "tunnel_num", "ct", "cl", "eff", "avgE"]]
BC_model_eff = BC_data_eff[["index", "in_index", "sensor_data_num", "tunnel_num", "ct", "cl", "eff", "avgE"]]
BC_model_fy = BC_data_fy[["index", "in_index", "sensor_data_num", "tunnel_num", "ct", "cl", "eff", "avgE"]]
expert_model = expert_model_df1[["index", "in_index", "sensor_data_num", "tunnel_num", "ct", "cl", "eff", "avgE"]]

# 保存到新文件
# transition_model.to_csv("transition_model_data.csv", index=False)
# BC_model_eff.to_csv("BC_model_eff.csv", index=False)
# BC_model_fy.to_csv("BC_model_fy.csv", index=False)
# expert_model.to_csv("expert_model_data_fy.csv", index=False)

single = [
    [0.210925715, -0.024391354, 0.310042793, 3.409002939],
    [0.232088507, 0.136137759, 0.296302143, 4.223846307],
    [0.189288311, 0.035222475, 0.299808273, 2.564302214],
    [0.184774475, -0.04424426, 0.230450065, 3.313233497],
    [0.208680404, 0.005242043, 0.313573873, 2.066207913],
    [0.259420809, 0.115820791, 0.345537089, 2.608999784],
    [0.331766068, 0.122883873, 0.304605387, 4.570422798],
    [0.281633094, -0.01390094, 0.240363132, 4.348727594]
]

whole_a = [
    [0.230683794, -0.075428695, 0.29635799, 3.648891161],
    [0.269283542, 0.025945935, 0.243055569, 4.702972731],
    [0.185006475, -0.066721746, 0.341296486, 2.591271951],
    [0.185167399, -0.120460815, 0.193302169, 3.693004584],
    [0.228697856, -0.08452704, 0.275813273, 2.325059754],
    [0.232235024, -0.008843962, 0.295035584, 2.707544652],
    [0.283146507, -0.046202209, 0.218622748, 4.281794698],
    [0.311524148, -0.07932536, 0.203655863, 4.598753291]
]

# 提取 ct, cl, eff 数据
# single_ct = [row[0] for row in single]
# single_cl = [row[1] for row in single]
# single_eff = [row[2] for row in single]

# whole_a_ct = [row[0] for row in whole_a]
# whole_a_cl = [row[1] for row in whole_a]
# whole_a_eff = [row[2] for row in whole_a]

offline_datas = pd.read_csv("inference_off2on.csv", header=None)
single_ct = []
whole_a_ct = []
bc_ct = []
cl_ct = []

single_cl = []
whole_a_cl = []
bc_cl = []
cl_cl = []

for i in range(len(offline_datas)):
    # s_data = offline_datas[i]
    s_data = offline_datas.iloc[i]
    ct = s_data[1]
    cl = abs(s_data[3])
    st = str(s_data[0])
    if 'off2on-1' in st:
        single_ct.append(ct)
        single_cl.append(cl)
    elif 'off2on-8' in st:
        whole_a_ct.append(ct)
        whole_a_cl.append(cl)
    elif 'bc' in st:
        bc_ct.append(ct)
        bc_cl.append(cl)
    else:
        cl_ct.append(ct)
        cl_cl.append(cl)



# 绘制 ct vs cl 散点图，使用小点
plt.figure(figsize=(8, 6))
plt.scatter(merged_df["ct"], merged_df["cl"], alpha=0.6, s=5, label='BF')
plt.scatter(single_ct, single_cl, alpha=0.6, s=20, color='blue', label='single')
plt.scatter(whole_a_ct, whole_a_cl, alpha=0.6, s=20, color='orange', label='whole_a')
plt.scatter(bc_ct, bc_cl, alpha=0.6, s=20, color='yellow', label='bc')
plt.scatter(cl_ct, cl_cl, alpha=0.6, s=20, color='green', label='for_cl')
plt.xlabel("ct")
plt.ylabel("cl")
plt.xlim(left=0)  # 仅显示第一象限
plt.ylim(bottom=0)  # 仅显示第一象限
plt.title("Scatter Plot of ct vs cl (in_index=60, After Outlier Removal)")
plt.legend()  # 添加图例
plt.grid(True)
plt.show()

# 绘制 ct vs eff 散点图，使用小点
# plt.figure(figsize=(8, 6))
# plt.scatter(merged_df["ct"], merged_df["eff"], alpha=0.6, s=5, color='r', label='BF')
# plt.scatter(single_ct, single_eff, alpha=0.6, s=20, color='blue', label='single')
# plt.scatter(whole_a_ct, whole_a_eff, alpha=0.6, s=20, color='orange', label='whole_a')
# plt.xlabel("ct")
# plt.ylabel("eff")
# plt.xlim(left=0)  # 仅显示第一象限
# plt.ylim(bottom=0)  # 仅显示第一象限
# plt.title("Scatter Plot of ct vs eff (in_index=60, After Outlier Removal)")
# plt.legend()  # 添加图例
# plt.grid(True)
# plt.show()

# 绘制 ct vs eff 散点图（使用异常值处理后的数据）
# plt.figure(figsize=(8, 6))
# plt.scatter(transition_model_df1["ct"], transition_model_df1["eff"], alpha=0.6, s=5, color='r')
# plt.xlabel("ct")
# plt.ylabel("eff")
# plt.xlim(left=0)  # 仅显示第一象限
# plt.ylim(bottom=0)  # 仅显示第一象限
# plt.title("Scatter Plot of ct vs eff (Filtered & Outlier Removed, in_index=60)")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.scatter(BC_model_eff["ct"], BC_model_eff["eff"], alpha=0.6, s=5, color='r')
# plt.xlabel("ct")
# plt.ylabel("eff")
# plt.xlim(left=0)  # 仅显示第一象限
# plt.ylim(bottom=0)  # 仅显示第一象限
# plt.title("BC_model_eff")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.scatter(BC_model_fx["ct"], BC_model_fx["eff"], alpha=0.6, s=5, color='r')
# plt.xlabel("ct")
# plt.ylabel("eff")
# plt.xlim(left=0)  # 仅显示第一象限
# plt.ylim(bottom=0)  # 仅显示第一象限
# plt.title("BC_model_fx")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.scatter(expert_model["ct"], expert_model["eff"], alpha=0.6, s=5, color='r')
# plt.xlabel("ct")
# plt.ylabel("eff")
# plt.xlim(left=0)  # 仅显示第一象限
# plt.ylim(bottom=0)  # 仅显示第一象限
# plt.title("expert_model")
# plt.grid(True)
# plt.show()
