import os
import pandas as pd
import numpy as np
import pickle
import math
import time


data_folder = '/home/iff/BF_tail/20241222'

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

expert_num = 0
best_num = 0
positive_num=0

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

for No in range(2500):
    i = No
    obsname = 'obs_{}.csv'.format(i)
    actname = 'act_{}.csv'.format(i)
    sensorname = 'sensor_data_{}.csv'.format(i)

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

    #Choosing expert with respect to Ct and Cl
    expert_list = []
    best_list = []
    positive_list = []
    for iff in range(n_IFF):
        # if abs(weighted_fy[iff]) > 0.01 and weighted_fx[iff] > 0.05 and (weighted_fx[iff] + abs(weighted_fy[iff])) > 0.25:
        #     expert_list.append(iff)
        #     expert_num += 1
        # if abs(weighted_fy[iff]) > 0.08 and weighted_fx[iff] > 0.08 and (weighted_fx[iff] + abs(weighted_fy[iff])) > 0.45:
        #     best_list.append(iff)
        #     best_num += 1
        # if weighted_fx[iff] > 0.05:
        # best_list.append(iff)
        # best_num += 1
            # print('Superbest: ', No, iff)
        # if (abs(weighted_fy[iff]) > 0.25 and weighted_fx[iff]> -0.05) or weighted_fx[iff] > 0.2:
        flag = (weighted_fx[iff]) **2 + (weighted_fy[iff]) **2 /2
        if flag < 0.06:
            rand = np.random.random()
            if weighted_fx[iff] > -0.005:
                if rand < flag / 0.06 + 0.25:
                    positive_list.append(iff)
                    positive_num += 1
            else:
                if rand < flag / 0.06 + 0.06:
                    positive_list.append(iff)
                    positive_num += 1
        else:
            positive_list.append(iff)
            positive_num += 1
            
        if weighted_fx[iff] > 0.4 or (weighted_fx[iff] > 0.25 and -0.1 < weighted_fy[iff] < 0.1):
            expert_list.append(iff)
            expert_num += 1

        if weighted_fy[iff] > 0.3 and weighted_fx[iff] > 0.12:
            best_list.append(iff)
            best_num += 1


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
        # cl_counting.append(single_s[:, :, 4])
        # if len(cl_counting) > cl_running_len:
        #     cl_counting.pop(0)
        #
        # # list * [n_iff, 1], transpose and unsqueeze to keep dims
        # cl = np.average(np.array(cl_counting).transpose(1, 2, 0), axis=-1)
        # cl = np.expand_dims(cl, axis=-1)
        #
        # single_s = np.concatenate((single_s, velocity_to_expand * np.pi / 90.0), axis=-1)
        # single_s = np.concatenate((single_s, cl), axis=-1)
        # # Interactively update velocities
        # velocity_to_expand = action

        ##########################################################

        next_s = obs_array[j + 1, :].reshape(n_IFF, 1, 9)

        # cl_counting_next = cl_counting.copy()
        # cl_counting_next.append(next_s[:, :, 4])
        # if len(cl_counting_next) > cl_running_len:
        #     cl_counting_next.pop(0)
        #
        # # list * [n_iff, 1], transpose and unsqueeze to keep dims
        # cl_next = np.average(np.array(cl_counting_next).transpose(1, 2, 0), axis=-1)
        # cl_next = np.expand_dims(cl_next, axis=-1)
        #
        # next_s = np.concatenate((next_s, action * np.pi / 90.0), axis=-1)
        # next_s = np.concatenate((next_s, cl_next), axis=-1)
        # single_sa = np.concatenate((single_s, action), axis=2)
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

        fx_indices = [3 + i * 9 for i in range(n_IFF)]
        fx_ = obs_array[j + 1, fx_indices]

        fy_indices = [4 + i * 9 for i in range(n_IFF)]
        fy_ = obs_array[j + 1, fy_indices]

        reward = 1.5 * fx_ - 0.3 * np.abs(fy_)

        # data_dict['rewards'].append(reward)
        # data_dict['fx'].append(fx_)
        # data_dict['fy'].append(fy_)
        REWARDS[j, :] = reward
        FYS[j, :] = fy_
        FXS[j, :] = fx_

        # print(fx_, fy_)

        # if expert_list:
        #     temp_expert_data_dict['actions'].append(np.take(action_save, expert_list, axis=0))
        #     temp_expert_data_dict['states'].append(np.take(Obs, expert_list, axis=0))
        #     temp_expert_data_dict['states_next'].append(np.take(Next_Obs, expert_list, axis=0))
        #     temp_expert_data_dict['states_actions'].append(np.take(SA, expert_list, axis=0))
        #     temp_expert_data_dict['dones'].append(np.take(dones, expert_list, axis=0))
        #     temp_expert_data_dict['rewards'].append(np.take(reward, expert_list, axis=0))
        #     temp_expert_data_dict['fx'].append(np.take(fx_, expert_list, axis=0))
        #     temp_expert_data_dict['fy'].append(np.take(fy_, expert_list, axis=0))
        # if best_list:
        #     temp_best_data_dict['actions'].append(action_save[best_list, :])
        #     temp_best_data_dict['states'].append(Obs[best_list, :, :])
        #     temp_best_data_dict['states_next'].append(Next_Obs[best_list, :, :])
        #     temp_best_data_dict['states_actions'].append(SA[best_list, :, :])
        #     temp_best_data_dict['dones'].append(np.take(dones, best_list, axis=0))
        #     temp_best_data_dict['rewards'].append(np.take(reward, best_list, axis=0))
        #     temp_best_data_dict['fx'].append(np.take(fx_, best_list, axis=0))
        #     temp_best_data_dict['fy'].append(np.take(fy_, best_list, axis=0))

    # Need to process 'expert data dict' cz each ?_iff may be different in each sample.
    # data_dict['actions'].append(ACTIONS)
    # data_dict['states'].append(STATES)
    # data_dict['states_next'].append(NEXT_STATES)
    # data_dict['states_actions'].append(SAS)
    # data_dict['dones'].append(DONES)
    # data_dict['rewards'].append(REWARDS)
    # data_dict['fx'].append(FXS)
    # data_dict['fy'].append(FYS)

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
        if maxmax > max_obs:
            max_obs = maxmax
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


#
# print(data_dict)
#
pickle_path = 'offline_data/offline_data_tail_transition.pkl'
pickle_path_expert = 'offline_data/offline_data_tail_fx_new2.pkl'
pickle_path_best = 'offline_data/offline_data_tail_fy_new2.pkl'

with open(pickle_path, 'wb') as f:
    pickle.dump(data_dict, f)
# #
# with open(pickle_path_expert, 'wb') as f:
#     pickle.dump(expert_data_dict, f)

# with open(pickle_path_best, 'wb') as f:
#     pickle.dump(best_data_dict, f)

# with open(pickle_path_expert, 'wb') as f:
#     pickle.dump(expert_data_dict, f)
print(positive_num, expert_num, best_num)

print(f'Data successfully saved to {pickle_path}')
print('Final time (including saving pickle): ', time.time() - start)
