#!/usr/bin/env python
"""
主脚本：多线程持续控制8台独立电机（MMGP_OL0 ~ MMGP_OL7）
- 每个线程无限循环：等待 flag=0 → 执行 → 完成 → 继续等待
- 所有8个线程完成当前轮次后，主控触发统一数据保存
- 电机映射：OL_i → 电机 (i + 4) % 8
- 修复 ROS 节点初始化问题（单 env 共享）
"""
import sys
import os
import threading
import time
import numpy as np

src_wzy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src_wzy')
sys.path.append(src_wzy_path)

from env.IFF_env_2 import ServoControlEnv


class SharedMotorEnv:
    def __init__(self, mock_params):
        print("[MAIN] 初始化 ServoControlEnv（ROS 节点）...")
        self.env = ServoControlEnv(mock_params)
        self.env.load_midvalue(mock_params.mid_values)
        self.env.reset()
        self.current_action = np.zeros((8, 3), dtype=np.float32)
        self.lock = threading.Lock()

    def set_motor_action(self, motor_idx, pitch_value):
        with self.lock:
            self.current_action[motor_idx, 0] = pitch_value

    def get_action_copy(self):
        with self.lock:
            return self.current_action.copy()

    def step(self):
        return self.env.step(self.current_action)

    def save_data(self):
        """由主控调用，保存完整数据"""
        try:
            self.env.save(0, save_full_data=True)
            return True
        except Exception as e:
            print(f"[SAVE] 保存原始数据失败: {e}")
            return False

    def reset_action_buffer(self):
        """重置动作为零（用于轮次间清理）"""
        with self.lock:
            self.current_action.fill(0.0)

    @property
    def average_Ct(self):
        return getattr(self.env, 'average_Ct', [])

    @property
    def average_Cl(self):
        return getattr(self.env, 'average_Cl', [])


class SingleMotorController:
    def __init__(self, ol_folder, motor_channel, shared_env, cycles, control_frequency, completion_event):
        self.ol_folder = ol_folder
        self.motor_channel = motor_channel
        self.shared_env = shared_env
        self.cycles = cycles
        self.step_interval = 1.0 / control_frequency
        self.completion_event = completion_event  # 通知主控本线程完成
        self.thread_id = f"OL{os.path.basename(ol_folder)[-1]}"  # e.g., OL0

    def run(self):
        print(f"[THREAD {self.thread_id}] 启动，映射到电机 {self.motor_channel}")
        while True:
            flag_path = os.path.join(self.ol_folder, "flag.txt")
            control_path = os.path.join(self.ol_folder, "control.txt")

            # === 等待 flag.txt == '0' ===
            while True:
                try:
                    if os.path.exists(flag_path):
                        with open(flag_path, 'r') as f:
                            if f.read().strip() == '0':
                                break
                except Exception as e:
                    print(f"[THREAD {self.thread_id}] 读取 flag.txt 异常: {e}")
                time.sleep(0.01)

            print(f"[THREAD {self.thread_id}] 检测到 flag=0，开始读取 control.txt")

            # === 读取控制序列 ===
            try:
                with open(control_path, 'r') as f:
                    control_values = [float(line.strip()) for line in f if line.strip()]
                if not control_values:
                    raise ValueError("control.txt 为空")
            except Exception as e:
                print(f"[THREAD {self.thread_id}] control.txt 读取失败: {e}")
                continue  # 跳过本轮，继续等待

            print(f"[THREAD {self.thread_id}] 加载 {len(control_values)} 步控制序列")

            # === 执行控制 ===
            self._execute_control_sequence(control_values)

            # === 标记完成（flag=1）===
            try:
                with open(flag_path, 'w') as f:
                    f.write('1')
                print(f"[THREAD {self.thread_id}] 控制完成，flag 已置为 '1'")
            except Exception as e:
                print(f"[THREAD {self.thread_id}] 更新 flag.txt 失败: {e}")

            # === 通知主控：本线程完成 ===
            self.completion_event.set()
            print(f"[THREAD {self.thread_id}] 已发送完成信号")

            # === 重置本电机动作为0（可选，避免残留）===
            self.shared_env.set_motor_action(self.motor_channel, 0.0)

            # 不退出，继续下一轮等待

    def _execute_control_sequence(self, control_values):
        cycle_time = len(control_values) * self.step_interval
        total_time = self.cycles * cycle_time
        start_time = time.time()
        last_cycle = -1

        while (time.time() - start_time) < total_time:
            elapsed = time.time() - start_time
            current_cycle = int(elapsed // cycle_time)
            time_in_cycle = elapsed % cycle_time
            step_idx = min(int(time_in_cycle // self.step_interval), len(control_values) - 1)
            control_val = control_values[step_idx]

            if current_cycle != last_cycle:
                print(f"[THREAD {self.thread_id}] 开始周期 {current_cycle + 1}/{self.cycles}")
                last_cycle = current_cycle

            self.shared_env.set_motor_action(self.motor_channel, control_val)
            time.sleep(self.step_interval * 0.5)


class MotorControlManager:
    def __init__(self, base_path=".", cycles=3):
        self.base_path = base_path
        self.cycles = cycles
        self.control_frequency = 3000

        class MockParams:
            def __init__(self):
                self.n_iff = 8
                self.excution_time = 0.035
                self.interval = 10
                self.steady_time = 0.0
                self.control_frequency = 3000
                self.refresh_time = 18
                self.mid_values = [
                    186,180,180,175,180,180,
                    179,180,180,180,180,180,
                    193,180,180,177,180,180,
                    189,180,180,184,180,180
                ]
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

    def get_ol_folders(self):
        return [os.path.join(self.base_path, f"MMGP_OL{i}") for i in range(8)]

    def run_concurrent(self):
        ol_folders = self.get_ol_folders()
        for f in ol_folders:
            if not os.path.isdir(f):
                raise FileNotFoundError(f"必需文件夹缺失: {f}")

        # 初始化共享环境（ROS 节点）
        shared_env = SharedMotorEnv(self.mock_params)

        # 为每个线程创建完成事件
        completion_events = [threading.Event() for _ in range(8)]
        threads = []

        print("[MAIN] 启动8个控制线程（持续模式）...")
        for i, folder in enumerate(ol_folders):
            motor_ch = (i + 4) % 8
            controller = SingleMotorController(
                ol_folder=folder,
                motor_channel=motor_ch,
                shared_env=shared_env,
                cycles=self.cycles,
                control_frequency=self.control_frequency,
                completion_event=completion_events[i]
            )
            thread = threading.Thread(target=controller.run, daemon=True)
            threads.append(thread)
            thread.start()

        print("[MAIN] 主控制循环启动（频率: {} Hz）...".format(self.control_frequency))
        try:
            while True:
                # 执行底层控制
                shared_env.step()
                time.sleep(max(0, 1.0 / self.control_frequency - 0.0001))

                # 检查是否所有线程都完成
                if all(event.is_set() for event in completion_events):
                    print("[MAIN] 检测到所有8个线程完成当前轮次，触发统一保存...")

                    # # 保存原始数据
                    # if shared_env.save_data():
                    #     print("[SAVE] 原始数据保存成功")

                    # 计算并写入无量纲力
                    self._save_dimensionless_forces(shared_env, ol_folders)

                    # 重置事件和动作缓冲
                    for event in completion_events:
                        event.clear()
                    shared_env.reset_action_buffer()
                    print("[MAIN] 已重置事件和动作缓冲，等待下一轮 flag=0\n")

        except KeyboardInterrupt:
            print("\n[MAIN] 收到中断信号，退出程序")

    def _save_dimensionless_forces(self, shared_env, ol_folders):
        try:
            avg_fx = np.mean(shared_env.average_Ct) if len(shared_env.average_Ct) > 0 else 0.0
            avg_fy = np.mean(shared_env.average_Cl) if len(shared_env.average_Cl) > 0 else 0.0
            print(f"[SAVE] 整体平均力 - X: {avg_fx:.6f}, Y: {avg_fy:.6f}")

            density, area, velocity = 1000.0, 0.01, 0.08
            dyn_press = 0.5 * density * velocity**2
            ct = avg_fx / (dyn_press * area) if dyn_press * area != 0 else 0.0
            cl = avg_fy / (dyn_press * area) if dyn_press * area != 0 else 0.0

            for folder in ol_folders:
                data_y_path = os.path.join(folder, 'dataY.txt')
                with open(data_y_path, 'a') as f:
                    f.write(f"{ct:.4f},{cl:.4f}\n")
            print(f"[SAVE] 无量纲力已追加到所有 dataY.txt: Ct={ct:.4f}, Cl={cl:.4f}")
        except Exception as e:
            print(f"[SAVE] 保存无量纲力失败: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='.')
    parser.add_argument('--cycles', type=int, default=5)
    args = parser.parse_args()

    manager = MotorControlManager(base_path=args.base_path, cycles=args.cycles)
    manager.run_concurrent()


if __name__ == "__main__":
    main()