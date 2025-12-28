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
from env.IFF_env_BF import ServoControlEnv
import time
from datetime import datetime
from framework import utils
class AdvancedMotorControlManager:
    
    def __init__(self, base_path=".", total_duration=10.0):
        self.base_path = base_path
        self.total_duration = total_duration  # 总运行时间（秒）
        self.n_iff = 8
        self.iteration=0

        # 从配置文件加载参数
        config_dict = utils.load_config(os.path.join(src_wzy_path, "config", "wzy.yaml"))
        paras = utils.get_paras_from_dict(config_dict)

        class MockParams:
            def __init__(self, paras):
                self.n_iff = paras.n_iff
                self.excution_time = paras.excution_time
                self.interval = paras.interval
                self.steady_time = paras.steady_time
                self.control_frequency = 3000
                self.refresh_time = paras.refresh_time
                self.mid_values = paras.mid_values
                self.action_space = paras.action_space
                self.obs_space = paras.obs_space
                self.motor_velocity = paras.motor_velocity
                self.sample_rate = paras.sample_rate
                self.cutoff_freq = paras.cutoff_freq
                self.order = paras.order
                self.rl_train = paras.rl_train
                self.save = paras.save
                self.bf_directory = paras.bf_directory
                self.r_alpha = paras.r_alpha
                self.r_beta = paras.r_beta
                self.r_gamma = paras.r_gamma
                self.rl_directory = paras.rl_directory
                self.starting_index = paras.starting_index
                self.cl_list_len = paras.cl_list_len
                self.seed = paras.seed
        
        self.mock_params = MockParams(paras)
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
        print(f"🚀 所有 flag=0，启动同步控制（总时长: {self.total_duration}s）iteration:{self.iteration}")
        
        # 初始化环境
        env = ServoControlEnv(self.mock_params)
        env.load_midvalue(self.mock_params.mid_values)
        obs = env.reset()
        # time.sleep(10)

        # 预计算每个电机的周期（秒）
        motor_periods = [len(seq) * self.step_interval for seq in sequences]
        print("📈 各电机周期（秒）:", [f"{p:.3f}" for p in motor_periods])

        start_time = time.time()
        steps = 0

        try:
            while (elapsed := time.time() - start_time) < self.total_duration:
                # 为每个电机计算当前控制值
                action = np.zeros((self.n_iff, 1))  # 8 motors × 3 DOF
                
                for ol_index, seq in enumerate(sequences):
                    motor_idx = (ol_index + 4) % 8  # 映射
                    period = motor_periods[ol_index]
                    # 循环索引：基于 elapsed_time % period
                    local_time = elapsed % period
                    idx_in_seq = int(local_time // self.step_interval)
                    idx_in_seq = min(idx_in_seq, len(seq) - 1)  # 安全边界
                    control_val = seq[idx_in_seq]
                    action[motor_idx, 0] = control_val  # pitching only
                # >>> 新增：每 3000 步打印一次 stepping_angles（pitching 维度）<<<
                if steps % 5 == 0:
                    # stepping_angles shape: (8, 3), 我们只关心第 0 列 (pitching)
                    current_angles = env.stepping_angles[:, 0]  # (8,)
                    print(f"⏱️  T={elapsed:.2f}s | Stepping Angles (Pitching, °): " +
                          " | ".join([f"M{i}:{a:6.2f}" for i, a in enumerate(current_angles)]))


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

            # # 重置电机
            env.refresh(0.0, 0.0)
            self.iteration+=1            # 在长时间运行后重置环境，清空 obs_array 等累积数据
            # print("🔄 等待 10 秒...")
            # time.sleep(10)  # 等待 10 秒

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
        """
        安全保存力数据，避免因 obs_array 形状不一致导致崩溃
        保存8个电机的独立平均力和无量纲系数（核心数据），不强制保存完整轨迹
        """
        # --- 第一步：提取平均力（关键输出） ---
        env.save(self.iteration, save_full_data=True)
        try:
            if hasattr(env, 'average_Ct') and hasattr(env, 'average_Cl'):
                avg_fx_list = env.average_Ct  # 长度为8的列表，对应8个电机
                avg_fy_list = env.average_Cl  # 长度为8的列表，对应8个电机

                if len(avg_fx_list) == 0 or len(avg_fy_list) == 0:
                    print("⚠️ 力数据为空，跳过保存")
                    return

                if len(avg_fx_list) != 8 or len(avg_fy_list) != 8:
                    print(f"⚠️ 力数据长度不正确 (期望8，实际: {len(avg_fx_list)}, {len(avg_fy_list)})，跳过保存")
                    return

            else:
                print("⚠️ env 缺少 average_Ct/Cl，无法计算平均力")
                return
        except Exception as e:
            print(f"⚠️ 提取平均力失败: {e}")
            return

        # --- 第二步：计算8个电机的无量纲系数 ---
        density = 1000    # kg/m³
        area = 0.1 * 0.1  # m²
        velocity = 0.15   # m/s
        dynamic_pressure = 0.5 * density * velocity**2

        # 计算8个电机的无量纲系数
        ct_values = [fx / (dynamic_pressure * area) for fx in avg_fx_list]
        cl_values = [fy / (dynamic_pressure * area) for fy in avg_fy_list]

        print(f"📊 各电机无量纲力: Ct = {[f'{ct:.4f}' for ct in ct_values]}, Cl = {[f'{cl:.4f}' for cl in cl_values]}")

        # --- 第三步：根据映射关系保存到对应的 OL 文件夹 ---
        # 映射: OLn → motor = (n + 4) % 8
        # 所以 motor 4,5,6,7,0,1,2,3 对应 OL0,1,2,3,4,5,6,7
        # 即: OL0<-motor4, OL1<-motor5, OL2<-motor6, OL3<-motor7, OL4<-motor0, OL5<-motor1, OL6<-motor2, OL7<-motor3
        for ol_index in range(8):
            motor_idx = (ol_index + 4) % 8  # 计算对应的电机索引
            folder = os.path.join(base_path, f"MMGP_OL{ol_index}")
            os.makedirs(folder, exist_ok=True)  # 确保目录存在
            data_y_path = os.path.join(folder, 'dataY.txt')
            with open(data_y_path, 'w') as f:
                f.write(f"{ct_values[motor_idx]:.4f},{cl_values[motor_idx]:.4f}\n")

        print("💾 无量纲力数据已保存到所有 MMGP_OL*/dataY.txt (按映射关系)")

        

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
                    #print(len(sequences))
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
    parser.add_argument('--duration', type=float, default=6.0, help='总运行时间（秒，默认10.0）')
    args = parser.parse_args()

    print("⚙️ 初始化同步电机控制系统...")
    manager = AdvancedMotorControlManager(base_path=args.base_path, total_duration=args.duration)
    manager.execute_continuous()

if __name__ == "__main__":
    main()