import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

count=0
fx_list = np.zeros(20000)
fy_list = np.zeros(20000)
w_list = np.zeros(20000)

# fig, axs = plt.subplots(5, 1, figsize=(16,30))
# plt.rcParams.update({'font.size': 18})
fig, axs = plt.subplots(1, 1, figsize=(30,16))
plt.rcParams.update({'font.size': 18})

counts = np.zeros(5)
# for set in [411, 1412, 1525, 1591, 1755]:
for set in [0, 290, 1470, 1514, 1674, 1769]:
    csv_file_path = '/home/iff/BF_tail/20250124_newtail_bf/sensor_data_{}.csv'.format(set)
    act_file_path = '/home/iff/BF_tail/20250124_newtail_bf/act_{}.csv'.format(set)
    obs_file_path = '/home/iff/BF_tail/20250124_newtail_bf/obs_{}.csv'.format(set)
    # csv_file_path = 'RLdata/sensor_data_{}.csv'.format(set)
    # act_file_path = 'RLdata/act_{}.csv'.format(set)
    # obs_file_path = 'RLdata/obs_{}.csv'.format(set)

    # csv_file_path = 'D:\\20241204\\sensor_data_{}.csv'.format(set)
    # act_file_path = 'D:\\20241204\\act_{}.csv'.format(set)
    # obs_file_path = 'D:\\20241204\\obs_{}.csv'.format(set)
    df_read = pd.read_csv(csv_file_path, usecols=range(97))

    data_array = df_read.to_numpy()
    timestampss = data_array[:, 0]
    timestamps = timestampss.copy()
    for llll in range(timestamps.shape[0]-1):
        lll = llll+1
        timestamps[lll] = timestampss[lll] - timestampss[lll-1]
    
    actions = pd.read_csv(act_file_path, header=None).to_numpy()
    actions = actions * np.pi /180.0

    # print(timestamps.mean())
    ttime = timestampss[-1]
    timestamps8 = np.repeat(timestamps, 8).reshape(timestampss.shape[0], 8)
    # print(timestamps8[-1,:])
    # print(data_array)

    # print(data_array.shape)


    fx_columns_indices = [49 + i * 6 for i in range(8)]
    fx_data = data_array[:, fx_columns_indices]

    ffx_columns_indices = [1 + i * 6 for i in range(8)]
    ffx_data = data_array[:, ffx_columns_indices]

    fy_columns_indices = [2 + i * 6 for i in range(8)]
    fy_data = data_array[:, fy_columns_indices]

    # mz_columns_indices = [6 + i * 6 for i in range(8)]
    # mz_data = data_array[:, fy_columns_indices]
    obs = pd.read_csv(obs_file_path, header=None).to_numpy()
    mz_columns_indices = [6 + i * 9 for i in range(8)]
    mz_data = obs[:, mz_columns_indices]

    mz_data = mz_data[1:, :] * actions
    z_times = np.linspace(0, 9 * mz_data.shape[0], mz_data.shape[0])


    # plt.plot(fx_data)
    # plt.show()

    for i in range(8):
        # axs.plot(ffx_data[:, i], label=f'fx{i + 1}', alpha=0.75)
        axs.plot(z_times, mz_data[:, i], label=f'mz * act{i + 1}', alpha=0.75)

    axs.set_title(f'Cycle {set + 1} - Filtered Force Measurements')
    axs.set_xlabel('Time Steps')
    axs.set_ylabel('Force (N)')
    axs.grid(True)
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    weighted_fx = np.average(ffx_data, axis=0, weights=timestamps)
    weighted_fy = np.average(mz_data, axis=0)
    print(f'Cycle {0}: \n Filtered mean fx: {weighted_fx}, \n Filtered mean fy: {weighted_fy}')
    # print(f'Cycle {set + 1}: Filtered mean fy: {weighted_fy}')



plt.tight_layout()
plt.show()


