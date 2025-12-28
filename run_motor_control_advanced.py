#!/usr/bin/env python
"""
主脚本：持续运行MMGP_OL文件夹的电机控制系统
此脚本位于父目录，从src_wzy目录导入ServoControlEnv等模块
使用实时时间同步控制8台电机，等待flag.txt为'0'时执行control.txt中的控制值，
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
                self.control_frequency = 3000  
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
        
        self.mock_params = MockParams()
        self.step_interval = 1.0 / self.mock_params.control_frequency 
        
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
            
            # 使用环境执行控制值 - 现在使用时间同步方法
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
        使用时间同步方法执行控制，仅使用pitching自由度
        根据实际运行时间确定当前控制值，而非固定步数
        """
        # 计算单个控制周期的时间 (秒)
        cycle_time = len(control_values) * self.step_interval
        
        # 计算总控制时间
        total_time = self.cycles * cycle_time
        start_time = time.time()
        
        print(f"为 {ol_folder} 执行时间同步控制:")
        print(f"  控制值数量: {len(control_values)}")
        print(f"  控制频率: {self.mock_params.control_frequency} Hz (步长 = {self.step_interval:.3f}秒)")
        print(f"  单周期时间: {cycle_time:.3f}秒")
        print(f"  总控制时间: {total_time:.3f}秒 ({self.cycles} 个周期)")
        
        steps_executed = 0
        last_cycle_index = -1
        
        try:
            # 主控制循环 - 基于实际时间
            while (current_time := time.time()) < start_time + total_time:
                # 计算已过去时间
                elapsed_time = current_time - start_time
                
                # 计算当前在总周期中的位置
                current_cycle = int(elapsed_time // cycle_time)
                time_in_cycle = elapsed_time % cycle_time
                
                # 根据时间在周期中的位置确定控制索引
                step_index = int(time_in_cycle // self.step_interval)
                step_index = min(step_index, len(control_values) - 1)  # 边界保护
                
                # 检测新周期开始
                if current_cycle > last_cycle_index:
                    print(f"  开始周期 {current_cycle + 1}/{self.cycles} (时间: {elapsed_time:.3f}/{total_time:.3f}秒)")
                    last_cycle_index = current_cycle
                
                # 获取当前控制值
                control_val = control_values[step_index]
                
                # 为8台电机创建仅含pitching自由度的动作(第一维)
                action = np.zeros((self.n_iff, 3))  # 8个电机，每个3个动作
                for motor_idx in range(self.n_iff):
                    action[motor_idx, 0] = control_val  # 仅pitching自由度
                
                # 在环境中执行动作
                next_obs, reward, done, info = env.step(action)
                steps_executed += 1
                
                # 监控执行状态
                if steps_executed % 10 == 0:  # 每10步打印一次状态
                    print(f"    周期 {current_cycle + 1}, 步骤 {step_index + 1}/{len(control_values)}: "
                          f"控制值 = {control_val:.3f} (总时间: {elapsed_time:.3f}/{total_time:.3f}秒)")
                
                # 处理完成状态
                if any(done):
                    print(f"    警告: 部分电机在 {ol_folder} 报告完成状态，重置环境")
                    env.reset()
                
                # 精确等待至下一步时间点
                current_time = time.time()

            
            print(f"  控制完成: 执行了 {steps_executed} 步，覆盖 {self.cycles} 个完整周期")
            
        except KeyboardInterrupt:
            print("\n  警告: 控制过程中断 (用户中断)")
            raise
        except Exception as e:
            print(f"  错误: 控制执行异常: {str(e)}")
            raise
    
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
            
            # 无量纲化处理
            density = 1000    # 水的密度 kg/m³
            area = 0.1 * 0.1  # 参考面积 m² (假设0.1m x 0.1m)
            velocity = 0.08   # 拖曳速度 m/s
            
            dynamic_pressure = 0.5 * density * velocity**2
            ct_avg = avg_fx_overall / (dynamic_pressure * area)
            cl_avg = avg_fy_overall / (dynamic_pressure * area)
            
            # 保存无量纲化后的数据到dataY.txt
            data_y = f"{ct_avg:.4f},{cl_avg:.4f}\n"
            data_y_path = os.path.join(ol_folder, 'dataY.txt')
            
            with open(data_y_path, 'a') as f:
                f.write(data_y)

            print(f"  无量纲化数据已追加到 {data_y_path}: {data_y.strip()}")
            
        else:
            print(f"无法检索 {ol_folder} 的平均力数据")

    def execute_continuous(self):
        """持续循环执行8个电机的控制，等待标志为0然后执行，永不停止"""
        ol_folders = self.get_ol_folders()
        
        if not ol_folders:
            print("未找到OL文件夹")
            return
        
        print("开始持续循环执行电机控制，等待标志为'0'然后执行...")
        print(f"控制频率: {self.mock_params.control_frequency} Hz")
        print(f"每个周期步数: 动态确定 (基于control.txt)")
        print("按 Ctrl+C 停止程序")
        
        try:
            while True:  # 无限循环
                executed_count = 0
                
                for ol_folder in ol_folders:
                    success = self.process_single_folder_with_env(ol_folder)
                    if success:
                        executed_count += 1
                
                if executed_count == 0:
                    # 没有执行任何文件夹时短暂休眠避免CPU过载
                    time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n接收到中断信号，停止持续执行...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='运行MMGP_OL文件夹的电机控制系统')
    parser.add_argument('--base_path', type=str, default='.', 
                       help='包含MMGP_OL文件夹的基路径(默认为当前目录)')
    parser.add_argument('--cycles', type=int, default=5, help='重复控制值的循环次数')
    
    args = parser.parse_args()
    
    print("从父目录初始化高级电机控制系统...")
    print(f"基路径: {args.base_path}")
    print(f"循环次数: {args.cycles}")
    
    # 创建高级电机控制管理器
    manager = AdvancedMotorControlManager(base_path=args.base_path, cycles=args.cycles)
    
    manager.execute_continuous()
    print("高级电机控制持续执行已停止。")


if __name__ == "__main__":
    main()        