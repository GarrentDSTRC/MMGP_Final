#!/usr/bin/env python
"""
主脚本：同步多电机独立周期控制（最终版）
- 仅当所有 MMGP_OL0~OL7 的 flag.txt 均为 '0' 时启动
- 每个电机独立使用自己的 control.txt（长度可不同）
- 控制序列循环播放，由全局时间驱动索引
- 所有电机同步运行 total_duration 秒后停止
- 映射: OLn → motor = (n + 4) % 8
- 单线程，时间同步，执行完保存数据并设所有 flag='1'
"""

import sys
import os

src_wzy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src_wzy')
sys.path.append(src_wzy_path)

import numpy as np
from env.IFF_env_2 import ServoControlEnv
import time
from datetime import datetime

class AdvancedMotorControlManager:
    
    def __init__(self, base_path=".", total_duration=10.0):
        self.base_path = base_path
        self.total_duration = total_duration  # 总运行时间（秒）
        self.n_iff = 8

        class MockParams:
            def __init__(self):
                self.n_iff = 8
                self.excution_time = 0.035
                self.interval = 10
                self.steady_time = 0.0
                self.control_frequency = 3000  
                self.refresh_time = 18
                self.mid_values = [186, 180, 180, 175, 180, 180, 179, 180, 180, 180, 180, 180, 
                                   193, 180, 180, 177, 180, 180, 189, 180, 180, 184, 180, 180]
                self.action_space = 3
                self.obs_space = 14
                self.motor_velocity = 0.08
                self.sample_rate = 220
                self.cutoff_freq = 10.0
                self.order = 5
                self.rl_train = False
                self.save = True
                self.bf_directory = "BF_data"
                self.r_alpha = 1.2
                self.r_beta = 0.4
                self.r_gamma = 0.1
                self.rl_directory = "RLtest/TPPO"
                self.starting_index = 0
                self.cl_list_len = 20
                self.seed = 1
        
        self.mock_params = MockParams()
        self.step_interval = 1.0 / self.mock_params.control_frequency  # ≈0.333ms

    def get_ol_folders(self):
        """返回 [(0, path0), (1, path1), ..., (7, path7)]"""
        folders = []
        for i in range(8):
            path = os.path.join(self.base_path, f"MMGP_OL{i}")
            if not os.path.isdir(path):
                raise FileNotFoundError(f"❌ 缺少目录: {path}")
            folders.append((i, path))
        return folders

    def all_flags_are_zero(self, ol_list):
        """检查所有 flag.txt 是否都为 '0'"""
        for _, folder in ol_list:
            flag_path = os.path.join(folder, "flag.txt")
            if not os.path.exists(flag_path):
                return False
            with open(flag_path, 'r') as f:
                if f.read().strip() != '0':
                    return False
        return True

    def load_all_control_sequences(self, ol_list):
        """加载所有 control.txt，返回 [seq0, seq1, ..., seq7]"""
        sequences = []
        for i, folder in ol_list:
            ctrl_path = os.path.join(folder, "control.txt")
            if not os.path.exists(ctrl_path):
                raise FileNotFoundError(f"❌ {folder} 缺少 control.txt")
            with open(ctrl_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                try:
                    seq = [float(x) for x in lines]
                except ValueError as e:
                    raise ValueError(f"❌ {folder}/control.txt 格式错误: {e}")
                if len(seq) == 0:
                    raise ValueError(f"❌ {folder}/control.txt 为空")
                sequences.append(seq)
        return sequences

    def execute_synchronized_run(self, ol_list, sequences):
        """同步执行所有电机控制，运行 total_duration 秒"""
        print(f"🚀 所有 flag=0，启动同步控制（总时长: {self.total_duration}s）")
        
        # 初始化环境
        env = ServoControlEnv(self.mock_params)
        env.load_midvalue(self.mock_params.mid_values)
        obs = env.reset()

        # 预计算每个电机的周期（秒）
        motor_periods = [len(seq) * self.step_interval for seq in sequences]
        print("📈 各电机周期（秒）:", [f"{p:.3f}" for p in motor_periods])

        start_time = time.time()
        steps = 0

        try:
            while (elapsed := time.time() - start_time) < self.total_duration:
                # 为每个电机计算当前控制值
                action = np.zeros((self.n_iff, 3))  # 8 motors × 3 DOF
                
                for ol_index, seq in enumerate(sequences):
                    motor_idx = (ol_index + 4) % 8  # 映射
                    period = motor_periods[ol_index]
                    # 循环索引：基于 elapsed_time % period
                    local_time = elapsed % period
                    idx_in_seq = int(local_time // self.step_interval)
                    idx_in_seq = min(idx_in_seq, len(seq) - 1)  # 安全边界
                    control_val = seq[idx_in_seq]
                    action[motor_idx, 0] = control_val  # pitching only

                # 同步执行所有电机
                next_obs, reward, done, info = env.step(action)
                steps += 1

                # 精确时间推进
                next_target_time = start_time + (steps) * self.step_interval
                now = time.time()
                if now < next_target_time:
                    # 忙等待（高精度），或可替换为 time.sleep(微小值)
                    while time.time() < next_target_time:
                        pass

            print(f"✅ 同步控制完成：运行 {steps} 步，历时 {time.time() - start_time:.3f}s")

            # 保存数据
            self.save_and_print_force_data(env, self.base_path)

            # 重置电机
            env.refresh(0.0, 0.0)

            # 更新所有 flag 为 '1'
            for _, folder in ol_list:
                flag_path = os.path.join(folder, "flag.txt")
                with open(flag_path, 'w') as f:
                    f.write('1')
            print("📝 所有 flag.txt 已更新为 '1'")

        except Exception as e:
            print(f"💥 执行出错: {e}")
            import traceback
            traceback.print_exc()
            # 尝试安全恢复
            try:
                env.refresh(0.0, 0.0)
            except:
                pass
            raise

    def save_and_print_force_data(self, env, base_path):
        """保存力数据（复用原逻辑）"""
        env.save(0, save_full_data=True)

        if hasattr(env, 'average_Ct') and hasattr(env, 'average_Cl'):
            avg_fx = np.mean(env.average_Ct) if len(env.average_Ct) > 0 else 0
            avg_fy = np.mean(env.average_Cl) if len(env.average_Cl) > 0 else 0

            density = 1000    # kg/m³
            area = 0.1 * 0.1  # m²
            velocity = 0.08   # m/s
            dynamic_pressure = 0.5 * density * velocity**2
            ct_avg = avg_fx / (dynamic_pressure * area)
            cl_avg = avg_fy / (dynamic_pressure * area)

            # 保存到每个 OL 文件夹（或统一目录？按您原逻辑追加到各 folder）
            for i in range(8):
                folder = os.path.join(base_path, f"MMGP_OL{i}")
                data_y_path = os.path.join(folder, 'dataY.txt')
                with open(data_y_path, 'a') as f:
                    f.write(f"{ct_avg:.4f},{cl_avg:.4f}\n")
            print(f"📊 平均无量纲力: Ct={ct_avg:.4f}, Cl={cl_avg:.4f}（已追加到各 dataY.txt）")
        else:
            print("⚠️ 无法获取平均力数据")

    def execute_continuous(self):
        """主循环：持续监听，等待全零 flag 后同步执行"""
        ol_list = self.get_ol_folders()
        print("🔄 进入监听模式：等待所有 flag.txt 变为 '0'...")
        print("   映射: OL0→4, OL1→5, OL2→6, OL3→7, OL4→0, OL5→1, OL6→2, OL7→3")
        print(f"   总运行时间: {self.total_duration} 秒")

        while True:
            if self.all_flags_are_zero(ol_list):
                try:
                    sequences = self.load_all_control_sequences(ol_list)
                    self.execute_synchronized_run(ol_list, sequences)
                except Exception as e:
                    print(f"❌ 同步执行失败: {e}")
                # 执行完后继续监听
            else:
                time.sleep(0.1)  # 降低 CPU 占用

def main():
    import argparse
    parser = argparse.ArgumentParser(description='同步多电机独立周期控制系统')
    parser.add_argument('--base_path', type=str, default='.', help='基路径（默认当前目录）')
    parser.add_argument('--duration', type=float, default=10.0, help='总运行时间（秒，默认10.0）')
    args = parser.parse_args()

    print("⚙️ 初始化同步电机控制系统...")
    manager = AdvancedMotorControlManager(base_path=args.base_path, total_duration=args.duration)
    manager.execute_continuous()

if __name__ == "__main__":
    main()