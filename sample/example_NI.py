import nidaqmx
import numpy as np
import time
import os
def main_sample(duration, T):
    # 初始化设备
    system = nidaqmx.system.System.local()
    device_count = len(system.devices)
    if device_count == 0:
        print("No device detected!")
        return []
    
    device_name = system.devices[0].name
    print(f"Device {device_name} detected.")

    # 采样参数配置
    frequency = 500  # 采样频率 (Hz)
    channels = list(range(8))  # 通道列表 0-7
    cycles = int(500)  # 总采样点数（每个通道）
    
    collected_data = []
    start_time = time.time()

    while time.time() - start_time < duration:
        with nidaqmx.Task() as task:
            # 配置通道
            task.ai_channels.add_ai_voltage_chan(
                f"{device_name}/ai0:7"  # 添加所有8个通道
            )

            # 设置采样参数
            task.timing.cfg_samp_clk_timing(
                rate=frequency,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_channel=cycles
            )

            # 配置软件触发
            task.triggers.start_trigger.cfg_software_start_trig()

            # 启动任务并触发
            task.start()
            task.start_trigger()  # 发送软件触发信号

            # 读取数据
            try:
                data = task.read(
                    number_of_samples_per_channel=cycles,
                    timeout=2
                )
            except nidaqmx.DaqError as e:
                print(f"Error reading data: {e}")
                continue

            # 数据处理（保持与原代码一致的结构）
            data_array = np.array(data)
            if np.max(data_array[0,:])>4 or np.max(data_array[0,:])<-4:
                os._exit()
            print("oringal sample",data_array[:,-1])

            collected_data.append(data_array)
            print(f"Collected {cycles} samples per channel")

    # 合并所有采集数据
    if not collected_data:
        return []
    
    # 沿时间轴拼接数据（每个通道的数据连续）
    all_data = np.concatenate(collected_data, axis=1)
    
    # 转置为通道在列的格式（与原代码一致）
    transposed_data = all_data.T  # 转置后形状为 (总采样点数, 8通道)
    
    # 转换为列表格式返回
    return transposed_data.tolist()