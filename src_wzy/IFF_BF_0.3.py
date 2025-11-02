import numpy as np
import csv
import os
from datetime import datetime
import math
import time
from framework.trainer1 import RSAC, TPPO  # , Trainer, SAC
from framework import utils
from framework.normalization import RewardScaling, Normalization
from model.sin_policy import SinPolicy
from datetime import datetime, timedelta
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import gym
import pickle
import torch
import rospy
import argparse
import json
from model.online_gpt_model import GPTConfig, GPT
from framework.utils import set_seed, ConfigDict, make_logpath
from framework.logger import LogServer, LogClient

from env.IFF_env_BF import ServoControlEnv




class ServoController:
    def __init__(self):
        self.angle1_accumulator = 0.0
        self.angle2_accumulator = 0.0
        self.old1 = 0.0
        self.old2 = 0.0

        self.AA = np.zeros((self.n_iff))

    def reset(self):
        self.angle1_accumulator = 0.0
        self.angle2_accumulator = 0.0
        self.old1 = 0.0
        self.old2 = 0.0

    def load_paras(self, input_list):
        #Input list: n_iff * 3-dim(AAxyz, AOxyz, APxyz, bias_xyz)
        pass

    def sin_func(self, AA, AO, AP, bias, t):
        #Paralleled by numpy arrays
        return AA * np.sin(AO * t + AP) + bias

    def servo_control(self):
        omega = 5 * math.pi / 5
        offset = 180
        amplitude = 30

        self.angle1_accumulator += 10
        self.angle2_accumulator += 10

        # 计算新的角度值
        theta1 = 10 * math.sin(math.radians(self.angle1_accumulator))
        theta2 = (10 + 5) * math.sin(math.radians(self.angle2_accumulator))

        angle1 = float(theta1 - self.old1)
        angle2 = float(theta1 - self.old1)
        angle3 = float(theta2 - self.old2)

        self.old1 = theta1
        self.old2 = theta2

        action = np.array([[angle1, angle2, angle3], [angle1, angle2, angle3]])

        return action

class Multi_ServoController:
    def __init__(self, n_iff):
        self.n_iff = n_iff
        self.old_angles = np.zeros((self.n_iff, 1))

        self.Amp = np.zeros((self.n_iff, 1))
        self.omega = np.zeros((self.n_iff, 1))
        self.phase = np.zeros((self.n_iff, 1))
        self.offset = np.zeros((self.n_iff, 1))

    def reset(self):
        self.old_angles = np.zeros((self.n_iff, 1))

    def load_paras(self, input_list):
        #Input list: n_iff * 3-dim(AAxyz, AOxyz, APxyz, bias_xyz)
        for i in range(self.n_iff):
            single_amp_set = input_list[i]['amp']
            self.Amp[i, 0], self.Amp[i, 1], self.Amp[i, 2] = single_amp_set[0], single_amp_set[1], single_amp_set[2]
            omega = input_list[i]['omega'][0] * 2.0 * math.pi
            self.omega[i, 0], self.omega[i, 1], self.omega[i, 2] = omega, omega, omega

    def load_paras_np(self, input_array):
        # columns=['omega1', 'omega2', 'omega3', 'theta0', 'theta_bar1', 'theta_bar2', 'theta_bar3', 'alpha2', 'alpha3']
        self.omega = np.expand_dims(input_array[:, 0], axis=1)
        self.offset = np.expand_dims(input_array[:, 1], axis=1)
        # self.phase = np.expand_dims(input_array[:, -1], axis=1)
        self.Amp = np.expand_dims(input_array[:, 2], axis=1)
        # for channel in range(8):
        #     limit = 60 - np.abs(self.offset[channel])
        #     if self.Amp[channel, 0] > limit:
        #         self.Amp[channel, 0] = limit
        # print(f'MSC loaded:, {self.omega} \n {self.offset} \n {self.Amp}')

    def sin_func(self, AA, AO, AP, bias, t):
        #Paralleled by numpy arrays
        return AA * np.sin(AO * t + AP)

    def servo_control(self, t):
        #Delta angle as actions
        new_angles = self.sin_func(self.Amp, self.omega, self.phase, self.offset, t)
        action = new_angles - self.old_angles
        self.old_angles = new_angles
        # s_action = np.expand_dims(action[:, 0], axis=1)

        return action

def prepare_arguments():
    parser = argparse.ArgumentParser()
    # Required_parameter
    parser.add_argument("--config-file", "--cf", default="./config/config.json",
                        help="pointer to the configuration file of the experiment", type=str)
    args, unknown = parser.parse_known_args()
    args.config = json.load(open(args.config_file, 'r', encoding='utf-8'))
    print(args.config)

    ### set seed
    if args.config['seed'] == "none":
        args.config['seed'] = datetime.now().microsecond % 65536
        args.seed = args.config['seed']
    set_seed(args.seed)

    # reconfig some parameter
    args.name = f"[Debug]FishRL_PPO_transformer_{args.seed}"
    # v4 actionDevide_eta_actionRange

    # wandb remote logger, can mute when debug
    mute = True
    remote_logger = LogServer(args, mute=mute)  # open logging when finish debuging
    remote_logger = LogClient(remote_logger)

    # for the hyperparameter search
    if mute:
        new_args = args
    else:
        new_args = remote_logger.server.logger.config if not mute else args
        new_args = ConfigDict(new_args)
        new_args.washDictChange()
    new_args.remote_logger = remote_logger
    return new_args, remote_logger


### load config AND prepare logger
args, remote_logger = prepare_arguments()
config = args.config
dir = "config/tppo.yaml"
config_dict = utils.load_config(dir)
paras = utils.get_paras_from_dict(config_dict)
print("local:", paras)
# wandb.init(project="Fish_0715", entity="krhkk")
wandb.init(project="Fish_0816", entity="krhkk", config=paras, name=args.name, mode="disabled")# ("disabled" or "online")
paras = utils.get_paras_from_dict(wandb.config)

# paras = utils.get_paras_from_dict(paras)

print("finetune", paras)
run_dir, log_dir = make_logpath('fish', paras.algo)
### start env
#env = foil_env(paras)
# num_envs = 10
num_envs = 4
# env = gym.vector.make('foil-v0', num_envs=num_envs, config=paras)
# env = gym.vector.make('fish-v0', num_envs=num_envs, config=paras, local_port=8686)
# env = foil_env(paras, local_port=8686)
# obs = env.reset()
#paras.action_space, action_dim = env.envs[0].action_dim, env.envs[0].action_dim
#paras.obs_space, observation_dim = env.envs[0].observation_dim, env.envs[0].observation_dim
# paras.action_space, action_dim = env.single_action_space.shape[0], env.single_action_space.shape[0]
# paras.obs_space, observation_dim = env.single_observation_space.shape[0], env.single_observation_space.shape[0]
paras.action_space, action_dim = 1, 1
paras.obs_space, observation_dim = 9, 9
paras.device = "cuda:0" if torch.cuda.is_available() else "cpu"
paras.env_num = 8


if __name__ == '__main__':
    rlist = []
    slist = []
    dlist = []
    snlist = []

    #Collect paras
    full_para = []

    starting_index = paras.starting_index

    csv_file_path = 'EMOA_tail_fxy_1-1.csv'
    # csv_file_path = 'BF_tail_fxbest.csv'
    df_read = pd.read_csv(csv_file_path)
    hyperpara_array = df_read.to_numpy()
    # rp = np.concatenate((hyperpara_array, hyperpara_array, hyperpara_array), axis=0)
    # hyperpara_array = rp

    mid_values = paras.mid_values

    MSC = Multi_ServoController(paras.n_iff)
    env = ServoControlEnv(paras)
    env.load_midvalue(mid_values)

    #Filling paras
    # left = hyperpara_array.shape[0] % paras.n_iff
    # if left > 0:
    #     hyperpara_array = hyperpara_array[:-left]

    #TODO: here specific for 4 channels
    # extend = hyperpara_array[0:4, :]

    # hyperpara_array = np.append(hyperpara_array, extend, axis=0)
    print('hyperpara array shape:', hyperpara_array.shape)

    # para_len = int(hyperpara_array.shape[0] // paras.n_iff)
    para_len = int(hyperpara_array.shape[0])

    environment_steps = int((1/paras.motor_velocity - paras.steady_time) * paras.control_frequency)
    print('BF Search: {} sets to go; Steps per Episode: {}'.format(para_len-starting_index, environment_steps))

    #BF Searching
    for i in range(starting_index, para_len):
        # starting = int(i * paras.n_iff)
        # local = hyperpara_array[starting: starting + paras.n_iff, :]

        local = np.expand_dims(hyperpara_array[i, :], axis=0)
        local = np.tile(local, (8, 1))
        print(local)

        MSC.load_paras_np(local)
        MSC.reset()
        time.sleep(0.5)
        env.adjust_midvalue(MSC.offset)
        time_counter = time.time()
        obs = env.reset()

        steps = 0.0
        dt = 1.0 / paras.control_frequency
        start_time = time.time()
        # remaining_time = 6.5 - (start_time - time_counter)
        remaining_time = 6
        execution_time = time.time() - start_time

        while not rospy.is_shutdown() and steps < environment_steps and execution_time < remaining_time:
            #Preprocess

            # execution_time = steps * dt
            execution_time = time.time() - start_time
            time.sleep(0.003)
            action = MSC.servo_control(execution_time)
            # print('act:', np.sum(action))
        
            # action = np.zeros((8, 1))
            # print('excution time:', execution_time)

            next_obs, r, d, _ = env.step(action)

            # time33 = time.time()
            # next_state = next_obs.reshape(next_obs.shape[0], 1, next_obs.shape[1])
            # next_state = np.concatenate([agent.state.cpu().detach().numpy(), next_state], axis=1)[:, 1:, :]
            # agent.insert_data({'states': agent.state.cpu().detach().numpy(), 'actions': action, 'rewards': r,
            #                    'states_next': next_state, 'dones': d})
            # obs = next_obs
            # print('delay3,', time.time() - time00)
            env.rate.sleep()
            steps += 1.0

        # env.tank.stop()

        # env.angles_msg.angles = env.mid_values.flatten().tolist()
        # env.pub.publish(env.angles_msg)
        # time.sleep(env.refresh_time)
        env.save(i, save_full_data=True)
        print("CTs: ", env.average_Ct)
        env.refresh(0.0, 0.0)
        # set += 1
 



