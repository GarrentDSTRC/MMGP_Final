#!/usr/bin/env python
"""
主脚本：持续运行MMGP_OL文件夹的电机控制系统
此脚本位于父目录，从src_wzy目录导入ServoControlEnv等模块
持续循环监控并控制8台电机，等待flag.txt为'0'时执行control.txt中的控制值，
仅控制pitching自由度，收集大地坐标系的力数据，保存到CSV，并打印平均力
"""
import sys
import os

# 将src_wzy目录添加到路径，以便导入所需模块
src_wzy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src_wzy')
sys.path.append(src_wzy_path)

import numpy as np
from env.IFF_env_2 import ServoControlEnv  # 导入伺服控制环境
import time
import csv
from datetime import datetime

class AdvancedMotorControlManager:
    """
    高级电机控制管理器，与ServoControlEnv接口进行8电机控制，
    持续循环监控并控制8台电机，等待flag.txt为'0'时执行，
    仅关注pitching自由度，收集力数据并保存到CSV
    """
    
    def __init__(self, base_path=".", cycles=3, n_iff=8):
        self.base_path = base_path
        self.cycles = cycles  # 控制循环次数
        self.n_iff = n_iff  # 电机数量(8个)
        
        # 用于初始化ServoControlEnv的模拟参数
        class MockParams:
            def __init__(self):
                self.n_iff = 8  # 8台电机
                self.excution_time = 0.035  # 执行时间
                self.interval = 10  # 控制间隔
                self.steady_time = 0.0  # 稳定时间
                self.control_frequency = 20  # 控制频率20Hz
                self.refresh_time = 18  # 刷新时间
                # 8台电机的初始中值 - 每个电机3个自由度 (pitching, heaving, swaying)
                self.mid_values = [186, 180, 180, 175, 180, 180, 179, 180, 180, 180, 180, 180, 193, 180, 180, 177, 180, 180, 189, 180, 180, 184, 180, 180]  # 8 motors * 3 values each
                self.action_space = 3  # 动作空间维度
                self.obs_space = 14  # 观测空间维度
                self.motor_velocity = 0.08  # 电机速度，设置为拖曳速度0.08m/s
                self.sample_rate = 220  # 采样率
                self.cutoff_freq = 10.0  # 截止频率
                self.order = 5  # 滤波器阶数
                self.rl_train = False  # 强化学习训练标志 - set to False to avoid RL directory issues
                self.save = True  # 保存标志
                self.bf_directory = "BF_data"  # 保存目录
                self.r_alpha = 1.2  # 参数α
                self.r_beta = 0.4   # 参数β
                self.r_gamma = 0.1  # 参数γ
                self.rl_directory = "RLtest/TPPO"  # RL directory (needed when rl_train is True)
                self.starting_index = 0  # Starting index for BF execution
                self.cl_list_len = 20  # Length for running CL list
                self.seed = 1  # Random seed
                # 只使用pitching自由度 - 动作的第一维
                # 设置其他维度为0，避免非pitching运动
        
        self.mock_params = MockParams()
        
    def get_ol_folders(self):
        """获取所有MMGP_OL文件夹列表"""
        ol_folders = []
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            # 检查是否为MMGP_OL开头的目录
            if os.path.isdir(item_path) and item.startswith("MMGP_OL") and item[7:].isdigit():
                ol_folders.append(item_path)
        
        ol_folders.sort()  # 排序确保顺序(OL0, OL1, OL2, 等)
        return ol_folders
    
    def process_single_folder_with_env(self, ol_folder):
        """
        使用ServoControlEnv处理单个OL文件夹的实际电机控制
        """
        control_path = os.path.join(ol_folder, "control.txt")  # 控制文件路径
        flag_path = os.path.join(ol_folder, "flag.txt")       # 标志文件路径
        
        # 检查flag.txt是否表明可以执行
        if not os.path.exists(flag_path):
            print(f"在 {ol_folder} 中找不到标志文件")
            return False
        
        # 持续等待直到标志为'0'
        flag_value = None
        while True:
            if os.path.exists(flag_path):
                with open(flag_path, 'r') as f:
                    flag_value = f.read().strip()
                
                if flag_value == '0':
                    break  # 标志为'0'，继续执行
                else:
                    print(f"{ol_folder} 中的标志是 '{flag_value}'，等待标志变为'0'...")
                    time.sleep(1)  # 等待1秒后再次检查
            else:
                print(f"在 {ol_folder} 中找不到标志文件，等待...")
                time.sleep(1)  # 等待1秒后再次检查
        
        # 检查控制文件是否存在
        if not os.path.exists(control_path):
            print(f"在 {ol_folder} 中找不到控制文件")
            return False
        
        # 从control.txt读取控制值
        with open(control_path, 'r') as f:
            try:
                control_values = [float(line.strip()) for line in f.readlines() if line.strip()]
            except ValueError:
                print(f"{ol_folder} 中control.txt的值无效")
                return False
        
        print(f"在 {ol_folder} 中找到 {len(control_values)} 个控制值，执行电机控制...")
        
        # 初始化ServoControlEnv进行实际电机控制
        try:
            env = ServoControlEnv(self.mock_params)  # 创建控制环境
            env.load_midvalue(self.mock_params.mid_values)  # 加载中值
            
            # 重置环境
            obs = env.reset()
            print(f"{ol_folder} 的环境已重置")
            
            # 使用环境执行控制值
            self.execute_control_with_env(env, control_values, ol_folder)
            
            # 保存数据并打印平均力
            self.save_and_print_force_data(env, ol_folder)
            
            # 执行重置过程，确保电机停止并恢复到安全状态
            env.refresh(0.0, 0.0)
            
            # 更新flag.txt为'1'表示执行完成
            with open(flag_path, 'w') as f:
                f.write('1')
            
            print(f"{ol_folder} 的电机控制完成，标志已更新为'1'")
            return True
            
        except Exception as e:
            print(f"{ol_folder} 执行电机控制时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def execute_control_with_env(self, env, control_values, ol_folder):
        """
        使用ServoControlEnv执行控制，仅使用pitching自由度
        """
        print(f"为 {ol_folder} 执行控制，包含 {len(control_values)} 个值和 {self.cycles} 个循环")
        
        # 使用时间间隔处理多循环的控制值
        for cycle in range(self.cycles):
            print(f"  为 {ol_folder} 开始循环 {cycle + 1}/{self.cycles}")
            
            # 使用适当的时间间隔(1/3000秒)处理每个控制值
            for i, control_val in enumerate(control_values):
                # 为8台电机创建仅含pitching自由度的动作(第一维)
                # 其他维度(第2、3维)设为0以避免非pitching运动
                action = np.zeros((self.n_iff, 3))  # 8个电机，每个3个动作
                
                # 将控制值应用到pitching自由度(第一维)
                for motor_idx in range(self.n_iff):
                    action[motor_idx, 0] = control_val  # 仅pitching自由度
                    # 其他维度保持为0(无heaving或swaying)
                
                # 在环境中执行动作
                next_obs, reward, done, info = env.step(action)
                
                print(f"    循环 {cycle + 1}, 步骤 {i + 1}/{len(control_values)}: "
                      f"对所有8个电机应用pitching控制值 {control_val:.3f}")
                
                # 如果任何电机完成，重置环境
                if any(done):
                    print(f"    一些电机完成，为 {ol_folder} 重置环境")
                    env.reset()
                
                # 简单休眠以控制时间
                # 在实际实现中，环境处理时间
                time.sleep(1.0/3000.0)
        
        print(f"  为 {ol_folder} 完成 {self.cycles} 个循环")
    
    def save_and_print_force_data(self, env, ol_folder):
        """
        保存力数据到CSV并打印平均力
        """
        # 使用环境的保存方法保存收集的数据
        env.save(0, save_full_data=True)
        
        # 打印x和y方向的平均力
        if hasattr(env, 'average_Ct') and hasattr(env, 'average_Cl'):
            avg_fx = env.average_Ct  # x方向的平均力
            avg_fy = env.average_Cl  # y方向的平均力
            
            print(f"{ol_folder} 的平均力:")
            print(f"  平均X力 (Ct): {avg_fx}")
            print(f"  平均Y力 (Cl): {avg_fy}")
            
            # 计算并打印整体平均值
            avg_fx_overall = np.mean(avg_fx) if len(avg_fx) > 0 else 0
            avg_fy_overall = np.mean(avg_fy) if len(avg_fy) > 0 else 0
            
            print(f"  整体平均X力: {avg_fx_overall:.4f}")
            print(f"  整体平均Y力: {avg_fy_overall:.4f}")
            
            # 无量纲化处理并保存到dataY.txt
            # 如果环境返回的已经是无量纲系数，则直接使用
            # 对于8个电机的平均值
            ct_avg = avg_fx_overall/ (0.5*  1000*  0.1*0.1 * 0.08**2)
            cl_avg = avg_fy_overall/ (0.5*  1000*  0.1*0.1 * 0.08**2)
            
            # 保存无量纲化后的数据到dataY.txt
            data_y = f"{ct_avg:.4f},{cl_avg:.4f}\n"

            # 写入到当前文件夹的dataY.txt文件
            data_y_path = os.path.join(ol_folder, 'dataY.txt')
            with open(data_y_path, 'a') as f:  # 使用追加模式，每次运行都添加一行
                f.write(data_y)

            print(f"  无量纲化数据已保存到 {data_y_path}: {data_y.strip()}")
            
        else:
            print(f"无法检索 {ol_folder} 的平均力数据")

    def execute_continuous(self):
        """持续循环执行8个电机的控制，等待标志为0然后执行，永不停止"""
        ol_folders = self.get_ol_folders()
        
        if not ol_folders:
            print("未找到OL文件夹")
            return
        
        print("开始持续循环执行电机控制，等待标志为'0'然后执行...")
        
        try:
            while True:  # 无限循环
                executed_count = 0
                
                for ol_folder in ol_folders:
                    success = self.process_single_folder_with_env(ol_folder)
                    if success:
                        executed_count += 1
                
                if executed_count > 0:
                    print(f"本次循环为 {executed_count} 个文件夹执行了电机控制")
                
                # 短暂休眠以避免过度占用CPU
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n接收到中断信号，停止持续执行...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='运行MMGP_OL文件夹的电机控制系统')
    parser.add_argument('--base_path', type=str, default='.', 
                       help='包含MMGP_OL文件夹的基路径(默认为当前目录)')
    parser.add_argument('--cycles', type=int, default=3, help='重复控制值的循环次数')
    
    args = parser.parse_args()
    
    print("从父目录初始化高级电机控制系统...")
    print(f"基路径: {args.base_path}")
    print(f"循环次数: {args.cycles}")
    
    # 创建高级电机控制管理器
    manager = AdvancedMotorControlManager(base_path=args.base_path, cycles=args.cycles)
    
    print("使用ServoControlEnv集成运行持续执行循环，等待标志为0然后执行，永不停止...")
    print("按 Ctrl+C 停止程序")
    manager.execute_continuous()
    print("高级电机控制持续执行已停止。")


if __name__ == "__main__":
    main()