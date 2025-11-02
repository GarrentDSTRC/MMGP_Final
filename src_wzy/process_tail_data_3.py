import os
import pandas as pd
import numpy as np
import pickle
import math
import time
import csv


# data_folder = '/home/iff/BF_tail/20250224_under30'
# data_folder = '/home/iff/BF_tail/20250124_newtail_bf'
###############################################################
# Generate actions with data c parameters & 'time' in data a
# load obs in data a
# for each single step, get s, a, sa, s_next, r, done, fx, fy
# merge s, sa, s_next in '[timesteps, n_IFF, context_len, dim]' fashion

###############################################################
# Attention! You should concate datas in experts and bests in another fashion;
# For example, just np.array can process full data dict 1d list * array(n_iff, context, dim) -> 4D array;
# But e&b dicts should be like 1d list * array(steps, ?_iff, context, dim)

n_IFF = 8
n_obs = 9
n_action = 1
context_len = 40

cl_running_len = 20


data_dict = {}
offline_properties = ['states', 'states_next', 'states_actions', 'rewards', 'dones', 'actions', 'returns', 'fx', 'fy']

expert_data_dict = {}
best_data_dict = {}

temp_expert_data_dict = {}
temp_best_data_dict = {}

for item in offline_properties:
    data_dict[item] = list()
    expert_data_dict[item] = list()
    best_data_dict[item] = list()
    temp_expert_data_dict[item] = list()
    temp_best_data_dict[item] = list()

start = time.time()
max_obs = 0.0

def process(id_list, channel_list, data_folder):
    expert_num = 0
    best_num = 0
    positive_num=0
    index = -1
    # for No in range(full_number):
    for No in id_list:
        i = No
        index += 1
        obsname = 'obs_{}.csv'.format(i)
        actname = 'act_{}.csv'.format(i)
        sensorname = 'sensor_data_{}.csv'.format(i)
        channel_iff = channel_list[index]

        file_obs = os.path.join(data_folder, obsname)
        file_act = os.path.join(data_folder, actname)
        file_sensor = os.path.join(data_folder, sensorname)

        df = pd.read_csv(file_obs, header=None)
        data = df.to_numpy()
        # for iff in range(n_IFF):
        #     data[:, iff * 9: iff * 9 + 3] = data[:, iff * 9: iff * 9 + 3]
            # data[:, iff * 9: iff * 9 + 3] = data[:, iff * 9: iff * 9 + 3] * math.pi / 180.0
        obs_array = data

        df = pd.read_csv(file_act, header=None)
        data = df.to_numpy()
        act_array = data

        df = pd.read_csv(file_sensor)
        data = df.to_numpy()
        data_array = data

        # timestamps = data_array[:, 0]
        timestampss = data_array[:, 0]
        timestamps = timestampss.copy()
        for llll in range(timestamps.shape[0]-1):
            lll = llll+1
            timestamps[lll] = timestampss[lll] - timestampss[lll-1]
        length = obs_array.shape[0] - 1
        # print('data array shape: ', data_array.shape)
        # print('obs shape: ', obs_array.shape)
        # print('act shape: ', act_array.shape)

        fx_columns_indices = [1 + i * 6 for i in range(n_IFF)]
        fx_data = data_array[:, fx_columns_indices]

        fy_columns_indices = [2 + i * 6 for i in range(n_IFF)]
        fy_data = data_array[:, fy_columns_indices]

        weighted_fx = np.average(fx_data, axis=0, weights=timestamps)
        weighted_fy = np.average(fy_data, axis=0, weights=timestamps)

        mz_id = [6 + i * 9 for i in range(n_IFF)]
        mz_data = obs_array[1:, mz_id]

        ang_id = [i * 9 for i in range(n_IFF)]
        ang_data = obs_array[1:, ang_id] - obs_array[:-1, ang_id]

        E_data = np.sum(mz_data * ang_data, axis=0) 

        eff = weighted_fx * 0.9 / (E_data + 1e-7)

        full_s_list = weighted_fx.flatten().tolist() + weighted_fy.flatten().tolist() + \
                      eff.flatten().tolist() + E_data.flatten().tolist()

        # countfile = 'offline_data/Avg_Fx_{}.csv'.format(full_number)
        # with open(countfile, mode='a', newline='') as file:
        #         writer = csv.writer(file)

        #         # writer.writerow(self.average_Ct)
        #         writer.writerow(full_s_list)
        if i == 0 or i==1:
            print(weighted_fx)

        #Choosing expert with respect to Ct and Cl
        expert_list = []
        best_list = []
        positive_list = []
        # for iff in range(n_IFF):
        #     # if abs(weighted_fy[iff]) > 0.01 and weighted_fx[iff] > 0.05 and (weighted_fx[iff] + abs(weighted_fy[iff])) > 0.25:
        #     #     expert_list.append(iff)
        #     #     expert_num += 1
        #     # if abs(weighted_fy[iff]) > 0.08 and weighted_fx[iff] > 0.08 and (weighted_fx[iff] + abs(weighted_fy[iff])) > 0.45:
        #     #     best_list.append(iff)
        #     #     best_num += 1
        #     # if weighted_fx[iff] > 0.05:
        #     # best_list.append(iff)
        #     # best_num += 1
        #         # print('Superbest: ', No, iff)
        #     # if (abs(weighted_fy[iff]) > 0.25 and weighted_fx[iff]> -0.05) or weighted_fx[iff] > 0.2:

        #     max_action = np.max(act_array[:, iff])

        #     x = weighted_fx[iff]
        #     y = eff[iff]

        #     if x>0 and 0< y < 1:
                
        #         if 0.05 < y < 0.085:
        #             value = np.random.random()
        #             if value < 0.61:
        #                 positive_list.append(iff)
        #                 positive_num += 1

        #         elif 0.025 < y <0.05:
        #             value = np.random.random()
        #             if value < 0.22:
        #                 positive_list.append(iff)
        #                 positive_num += 1
                
        #         elif 0.0 < y <0.025:
        #             value = np.random.random()
        #             if value < 0.06:
        #                 positive_list.append(iff)
        #                 positive_num += 1
                
        #         else:
        #             positive_list.append(iff)
        #             positive_num += 1
                
        #     # else:
        #     #     positive_list.append(iff)
        #     #     positive_num += 1


        #     # flag = (weighted_fx[iff]) **2 + (weighted_fy[iff]) **2 /2
        #     # if flag < 0.06:
        #     #     rand = np.random.random()
        #     #     if weighted_fx[iff] > -0.005:
        #     #         if rand < flag / 0.06 + 0.25:
        #     #             positive_list.append(iff)
        #     #             positive_num += 1
        #     #     else:
        #     #         if rand < flag / 0.06 + 0.06:
        #     #             positive_list.append(iff)
        #     #             positive_num += 1
        #     # else:
        #     #     positive_list.append(iff)
        #     #     positive_num += 1
                
        #     # if weighted_fx[iff] > 0.055 and weighted_fy[iff] > -0.07 :
        #     #     expert_list.append(iff)
        #     #     expert_num += 1

        #     # if weighted_fx[iff] > -0.05 and weighted_fy[iff] >0.10:
        #     #     best_list.append(iff)
        #     #     best_num += 1
            
        #     if (x > 0.08 and 0.01<y<1) or (0<x<0.08 and 0.06 < y <1):
        #         expert_list.append(iff)
        #         expert_num += 1

        #     if 0<x<0.8 and 0.08<y<1:
        #         best_list.append(iff)
        #         best_num += 1

        positive_list.append(channel_iff)
        positive_num += 1

        STATES = np.zeros([length, n_IFF, context_len, n_obs])
        NEXT_STATES = np.zeros([length, n_IFF, context_len, n_obs])
        ACTIONS = np.zeros([length, n_IFF, n_action])
        SAS = np.zeros([length, n_IFF, context_len, n_action])
        DONES = np.zeros([length, n_IFF])
        REWARDS = np.zeros([length, n_IFF])
        FXS = np.zeros([length, n_IFF])
        FYS = np.zeros([length, n_IFF])

        rolling_s = np.zeros([n_IFF, context_len, n_obs])
        rolling_sa = np.zeros([n_IFF, context_len, n_action])
        cl_counting = []

        dones = np.zeros([n_IFF])
        velocity_to_expand = np.zeros([n_IFF, 1, n_action])
        cl_running_avg = np.zeros([n_IFF, 1, 1])

        for j in range(length):
            # print(j)

            action = act_array[j, :].reshape(n_IFF, 1, 1)
            action_save = action.squeeze(1)
            # data_dict['actions'].append(action_save)
            ACTIONS[j, :, :] = action_save

            single_s = obs_array[j, :].reshape(n_IFF, 1, 9)
            ##########################################################

            next_s = obs_array[j + 1, :].reshape(n_IFF, 1, 9)
            single_sa = action

            # Merge them with history observations @ 50 context length
            Obs = np.concatenate((rolling_s[:, 1:, :], single_s), axis=1)
            # data_dict['states'].append(Obs)
            STATES[j, :, :, :] = Obs

            rolling_s = Obs
            # print(rolling_s)

            rolling_next_s = rolling_s
            Next_Obs = np.concatenate((rolling_next_s[:, 1:, :], next_s), axis=1)
            # data_dict['states_next'].append(Next_Obs)
            NEXT_STATES[j, :, :, :] = Next_Obs

            SA = np.concatenate((rolling_sa[:, 1:, :], single_sa), axis=1)
            # data_dict['states_actions'].append(SA)
            SAS[j, :, :, :] = SA
            rolling_sa = SA

            DONES[j, :] = dones
            # data_dict['dones'].append(dones)

            fx_indices = [1 + i * 9 for i in range(n_IFF)]
            fx_ = obs_array[j + 1, fx_indices]

            fy_indices = [2 + i * 9 for i in range(n_IFF)]
            fy_ = obs_array[j + 1, fy_indices]

            reward = 0.25 * fx_ + 1.5 * fy_

            REWARDS[j, :] = reward
            FYS[j, :] = fy_
            FXS[j, :] = fx_


        if expert_list:
            expert_data_dict['actions'].append(np.take(ACTIONS, expert_list, axis=1))
            expert_data_dict['states'].append(np.take(STATES, expert_list, axis=1))
            expert_data_dict['states_next'].append(np.take(NEXT_STATES, expert_list, axis=1))
            expert_data_dict['states_actions'].append(np.take(SAS, expert_list, axis=1))
            expert_data_dict['dones'].append(np.take(DONES, expert_list, axis=1))
            expert_data_dict['rewards'].append(np.take(REWARDS, expert_list, axis=1))
            expert_data_dict['fx'].append(np.take(FXS, expert_list, axis=1))
            expert_data_dict['fy'].append(np.take(FYS, expert_list, axis=1))
        if best_list:
            maxmax = np.max(np.take(STATES, best_list, axis=1))
            # if maxmax > max_obs:
            #     max_obs = maxmax
                # print(np.take(STATES, best_list, axis=1))
            best_data_dict['actions'].append(np.take(ACTIONS, best_list, axis=1))
            best_data_dict['states'].append(np.take(STATES, best_list, axis=1))
            best_data_dict['states_next'].append(np.take(NEXT_STATES, best_list, axis=1))
            best_data_dict['states_actions'].append(np.take(SAS, best_list, axis=1))
            best_data_dict['dones'].append(np.take(DONES, best_list, axis=1))
            best_data_dict['rewards'].append(np.take(REWARDS, best_list, axis=1))
            best_data_dict['fx'].append(np.take(FXS, best_list, axis=1))
            best_data_dict['fy'].append(np.take(FYS, best_list, axis=1))
        if positive_list:
            data_dict['actions'].append(np.take(ACTIONS, positive_list, axis=1))
            data_dict['states'].append(np.take(STATES, positive_list, axis=1))
            data_dict['states_next'].append(np.take(NEXT_STATES, positive_list, axis=1))
            data_dict['states_actions'].append(np.take(SAS, positive_list, axis=1))
            data_dict['dones'].append(np.take(DONES, positive_list, axis=1))
            data_dict['rewards'].append(np.take(REWARDS, positive_list, axis=1))
            data_dict['fx'].append(np.take(FXS, positive_list, axis=1))
            data_dict['fy'].append(np.take(FYS, positive_list, axis=1))
    
    return positive_num

path_60 = '/home/iff/BF_tail/20250220'
path_61 = '/home/iff/BF_tail/20250221'
path_30 = '/home/iff/BF_tail/20250224_under30'

pick = 'expert_model_data.csv'

alldata = pd.read_csv(pick).to_numpy()
chosen_indexes = [[], [], []] #60, 61, 30
channels = [[], [], []]
for ll in range(alldata.shape[0]):
    if alldata[ll, 0] == 60:
        chosen_indexes[0].append(int(alldata[ll, 2]))
        channels[0].append(int(alldata[ll, 3]))
    elif alldata[ll, 0] == 61:
        chosen_indexes[1].append(int(alldata[ll, 2]))
        channels[1].append(int(alldata[ll, 3]))
    elif alldata[ll, 0] == 30:
        chosen_indexes[2].append(int(alldata[ll, 2]))
        channels[2].append(int(alldata[ll, 3]))
print(chosen_indexes)
#
# print(data_dict)
#
full_number = 0
full_number += process(chosen_indexes[0], channels[0], '/home/iff/BF_tail/20250220')
full_number += process(chosen_indexes[1], channels[1], '/home/iff/BF_tail/20250221')
full_number += process(chosen_indexes[2], channels[2], '/home/iff/BF_tail/20250224_under30')

pickle_path = 'offline_data/offline_data_newKF_expert.pkl'
pickle_path_expert = 'offline_data/offline_data_newKF_expert.pkl'
pickle_path_best = 'offline_data/offline_data_newKF_eff.pkl'

with open(pickle_path, 'wb') as f:
    pickle.dump(data_dict, f)
#
# with open(pickle_path_expert, 'wb') as f:
#     pickle.dump(expert_data_dict, f)

# with open(pickle_path_best, 'wb') as f:
#     pickle.dump(best_data_dict, f)

# with open(pickle_path_expert, 'wb') as f:
#     pickle.dump(expert_data_dict, f)
print(f"positive: {full_number}")

print(f'Data successfully saved to {pickle_path}')
print('Final time (including saving pickle): ', time.time() - start)
