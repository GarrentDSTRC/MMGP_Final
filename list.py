import math
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import gpytorch
from matplotlib import pyplot as plt
from pyKriging.samplingplan import samplingplan
import pandas as pd
from time import time
from scipy.interpolate import griddata
import os
import numpy as np
import random
from GPy import *
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required functions from GPy.py
from GPy import normalizer, findpointOL, UPB, LOWB

class BatchDataProcessor:
    def __init__(self, centroids_path=r"./Database/centroids.csv", data_path=r"./Database/x2_modified_corrected.csv"):
        """
        初始化数据处理器，进行数据分组
        """
        # 读取质心数据以获取分组的唯一键值
        centroids_data = np.loadtxt(centroids_path, delimiter=',')
        # 分组的唯一键值是质心文件第一列的值
        self.unique_keys = centroids_data[:, 0]  # 来自质心csv文件的第一列值
        print("来自质心的唯一键值:", self.unique_keys)

        # 加载主数据
        self.data = np.loadtxt(data_path, delimiter=',')

        print("数据形状:", self.data.shape)
        print("前几行数据:", self.data[:3])

        # 根据第4列(索引3)使用质心中的键值对数据进行分组
        self.groups = {key: [] for key in self.unique_keys}

        # 对于数据中的每一行，根据第4列找到它所属的组
        for row in self.data:
            fourth_col = row[3]  # 第4列(索引3)
            # 找到最接近的质心键值以处理潜在的浮点数差异
            closest_key_idx = np.argmin(np.abs(self.unique_keys - fourth_col))
            closest_key = self.unique_keys[closest_key_idx]
            self.groups[closest_key].append(row)

        # 将列表转换为numpy数组
        for key in self.groups:
            self.groups[key] = np.array(self.groups[key])
            print(f"键值为 {key:.6f} 的组有 {len(self.groups[key])} 个项")

        # 按顺序获取组键值（与质心相同顺序）
        self.ordered_keys = list(self.unique_keys)
        
        # 计算最小长度（所有组中元素数最少的那个组的长度）
        self.min_len = min(len(self.groups[key]) for key in self.ordered_keys) if self.groups else 0
        self.current_pos = 0  # 当前位置
        
    def get_next_batch(self, batch_size=16):
        """
        按顺序获取下一个批次的数据
        返回批次数据和一个布尔值表示是否还有更多批次
        """
        if self.current_pos >= self.min_len:
            return None, False  # 没有更多批次
            
        batch_data = []
        
        # 如果存在，则取每组在位置'pos'的一个元素
        for key in self.ordered_keys:
            group = self.groups[key]
            if self.current_pos < len(group):
                batch_data.append(group[self.current_pos])
            else:
                # 如果此组在此位置没有元素，则返回None和结束标志
                print(f"组 {key} 在位置 {self.current_pos} 没有元素，停止循环")
                return None, False
        
        # 如果我们有一些数据，通过循环可用数据来达到指定批次大小
        if len(batch_data) > 0:
            # 通过循环可用数据创建指定大小的批次
            # 为了获得 5678123456781234 的顺序而不是 1234567812345678，
            # 我们需要从 batch_data 的一半位置开始循环
            full_batch = []
            start_offset = len(batch_data) // 2  # 从中间开始
            for i in range(batch_size):
                idx = (start_offset + i) % len(batch_data)
                full_batch.append(batch_data[idx])
            
            self.current_pos += 1  # 移动到下一个位置
            return np.array(full_batch), True
        else:
            self.current_pos += 1  # 移动到下一个位置
            return None, self.current_pos < self.min_len


# 使用新的类结构
processor = BatchDataProcessor()

# 准备输出
testmode = "experiment_cluster"  # 与原始代码相同
path2 = r"./Database/xy.csv"  # 修改文件名以反映内容
ALL_X = None
ALL_Y = None

# 按顺序获取批次并处理
pos = 0
while True:
    batch_X_normalized, has_more = processor.get_next_batch(batch_size=16)
    
    if batch_X_normalized is None:
        print(f"在位置 {pos} 处停止，因为不是所有组都有元素或没有更多批次")
        break
    
    # 反归一化得到实际物理输入 X
    X = normalizer.denormalize(batch_X_normalized)  # 确保 normalizer 已定义
    
    # 使用 findpointOL 处理，获取对应的 Y
    initialDataX, initialDataY = findpointOL(X, num_task=2, mode=testmode)
    
    # 注意：findpointOL 返回的 initialDataX 应该与 X 一致或为处理后的 X
    # 但根据您的需求，我们使用反归一化后的 X 作为输入特征
    # 因此我们直接使用 X（而非 initialDataX）作为保存的输入
    
    # 累积 X 和 Y
    if ALL_X is None:
        ALL_X = X
        ALL_Y = initialDataY
    else:
        ALL_X = np.concatenate((ALL_X, X), axis=0)
        ALL_Y = np.concatenate((ALL_Y, initialDataY), axis=0)

    print(f"处理位置 {pos} 的批次, 当前 X 形状: {ALL_X.shape}, Y 形状: {ALL_Y.shape}")
    pos += 1

    # 水平拼接 X 和 Y: [x1, x2, ..., xn, y1, y2, ...]
    if ALL_X is not None and ALL_Y is not None:
        XY_combined = np.hstack((ALL_X, ALL_Y))
        np.savetxt(path2, XY_combined, delimiter=',')
        print(f"最终 XY 拼接形状: {XY_combined.shape}")
        print(f"数据已保存到: {path2}")
    else:
        print("未生成任何数据，跳过保存。")

print("处理完成.")
