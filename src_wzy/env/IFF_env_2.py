from gym.spaces import Box
import os
import signal
import pickle
import rospy
import gym
from gym import spaces
import numpy as np
from ss.msg import SensorMsg, MotorAngles
from std_msgs.msg import Bool
import csv
import os
from datetime import datetime
import math
from math import pi
import time
from env.Tankmotor import Tankmotor
from env.lpftest import ZeroPhaseLowPassFilter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class ServoControlEnv(gym.Env):
    def __init__(self, args):
        super(ServoControlEnv, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.massive_filters = [ZeroPhaseLowPassFilter(args.sample_rate, args.cutoff_freq, args.order) for _ in range(48)]

        ##############################################################################
        # TIME AND DELAY CONTROLLING
        self.n_iff = args.n_iff
        self.action_interval = args.interval
        self.break_time = args.excution_time / self.action_interval
        self.steady_time = args.steady_time
        self.control_freq = args.control_frequency
        self.refresh_time = args.refresh_time
        self.action_dim = args.action_space
        self.obs_dim = args.obs_space
        self.dim = 3

        print('break time:', self.break_time)

        self.stepping_angles = np.ones([self.n_iff, self.action_dim]) * 180.0
        self.ini_mid_values = np.ones([self.n_iff, self.action_dim]) * 180.0
        self.mid_values = self.ini_mid_values
        self.lower_values = self.mid_values - 60.0
        self.upper_values = self.mid_values + 60.0

        ##############################################################################
        # LOCAL DATA SAVING
        home_directory = os.path.expanduser("~")

        if args.rl_train and args.save:
            bfdata_directory = os.path.join(home_directory, args.rl_directory)
            bfdata_directory = os.path.join(bfdata_directory, str(args.seed))
            if not os.path.exists(bfdata_directory):
                os.makedirs(bfdata_directory)
        else:
            bfdata_directory = os.path.join(home_directory, args.bf_directory)
            bfdata_directory = os.path.join(bfdata_directory, str(args.seed))
            if not os.path.exists(bfdata_directory):
                os.makedirs(bfdata_directory)

        self.csvname = os.path.join(bfdata_directory, "sensor_data_{}.csv")
        self.picklename = os.path.join(bfdata_directory, "sensor_data_{}.pkl")

        self.countname = os.path.join(bfdata_directory, "iter_count.txt")
        self.countfile = os.path.join(bfdata_directory, "AvgFx.csv")

        self.obs_template = os.path.join(bfdata_directory, "obs_{}.csv")
        self.act_template = os.path.join(bfdata_directory, "act_{}.csv")

        self.r_alpha = args.r_alpha
        self.r_beta  = args.r_beta
        self.r_gamma = args.r_gamma

        self.obs_array = []
        self.act_array = []

        self.current_sensor_data = []
        self.full_sensor_data = []
        self.collect_time = time.time()
        self.time_ls = []
        self.starting_time = time.time()

        self.average_Ct = []
        self.average_Cl = []

        self.running_cl = []
        self.cl_list_len = args.cl_list_len

        ##############################################################################
        # ROSPY INTERFACE
        rospy.init_node('controller', anonymous=True)

        self.pub = rospy.Publisher('motor_control_all', MotorAngles, queue_size=10)
        self.subscriber = rospy.Subscriber('sensor_data', SensorMsg, self.sensor_data_callback)
        #TODO: Check where to put the subscriber
        self.rate = rospy.Rate(self.control_freq)

        self.pub_zero = rospy.Publisher('set_zero', Bool, queue_size=10)
        self.pub_receive = rospy.Publisher('start_receiving', Bool, queue_size=10)

        self.tank = Tankmotor()

        self.angles_msg = MotorAngles()


    def start(self):
        rospy.spin()

    def save(self, iter, save_full_data=True, save_type='csv'):
        #TODO: Using wandb
        self.angles_msg.angles = self.ini_mid_values.flatten().tolist()
        self.pub.publish(self.angles_msg)

        if save_full_data:
            # Load sensor data immediately
            list_to_save = self.full_sensor_data
            if save_type=='csv':
                filename = self.csvname.format(iter)
                with open(filename, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(['timestamp',
                                         'Ffx1', 'Ffy1', 'Ffz1', 'Tfx1', 'Tfy1', 'Tfz1',
                                         'Ffx2', 'Ffy2', 'Ffz2', 'Tfx2', 'Tfy2', 'Tfz2',
                                         'Ffx3', 'Ffy3', 'Ffz3', 'Tfx3', 'Tfy3', 'Tfz3',
                                         'Ffx4', 'Ffy4', 'Ffz4', 'Tfx4', 'Tfy4', 'Tfz4',
                                         'Ffx5', 'Ffy5', 'Ffz5', 'Tfx5', 'Tfy5', 'Tfz5',
                                         'Ffx6', 'Ffy6', 'Ffz6', 'Tfx6', 'Tfy6', 'Tfz6',
                                         'Ffx7', 'Ffy7', 'Ffz7', 'Tfx7', 'Tfy7', 'Tfz7',
                                         'Ffx8', 'Ffy8', 'Ffz8', 'Tfx8', 'Tfy8', 'Tfz8',
                                         'Fx1', 'Fy1', 'Fz1', 'Tx1', 'Ty1', 'Tz1',
                                         'Fx2', 'Fy2', 'Fz2', 'Tx2', 'Ty2', 'Tz2',
                                         'Fx3', 'Fy3', 'Fz3', 'Tx3', 'Ty3', 'Tz3',
                                         'Fx4', 'Fy4', 'Fz4', 'Tx4', 'Ty4', 'Tz4',
                                         'Fx5', 'Fy5', 'Fz5', 'Tx5', 'Ty5', 'Tz5',
                                         'Fx6', 'Fy6', 'Fz6', 'Tx6', 'Ty6', 'Tz6',
                                         'Fx7', 'Fy7', 'Fz7', 'Tx7', 'Ty7', 'Tz7',
                                         'Fx8', 'Fy8', 'Fz8', 'Tx8', 'Ty8', 'Tz8',
                                         ])
                    csv_writer.writerows(list_to_save)

                obsname = self.obs_template.format(iter)
                with open(obsname, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(self.obs_array)

                actname = self.act_template.format(iter)
                with open(actname, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(self.act_array)

            else:
                pass

            # For quick evaluate
            data_array = np.array(list_to_save)
            print('Evaluating data',data_array.shape)
            steps = data_array.shape[0]

            np_obs = np.array(self.obs_array)
            # np_act = np.array(self.act_array) * np.pi / 180.0

            mz_indices = [6 + i * 9 for i in range(8)]
            ang_indices = [i * 9 for i in range(8)]
            np_act = np_obs[1:, ang_indices] - np_obs[:-1, ang_indices]
            # print(f"ACT: {np_act} \n 'OBS: {act}\n")
            mz = np_obs[1:, mz_indices]
            Energy = np.sum(np_act * mz, axis=0)

            timestampss = data_array[:, 0]
            timestamps = timestampss.copy()
            for llll in range(timestamps.shape[0] - 1):
                lll = llll + 1
                timestamps[lll] = timestampss[lll] - timestampss[lll-1]

            fx_columns_indices = [1 + i * 6 for i in range(self.n_iff)]
            fx_data = data_array[:, fx_columns_indices]

            fy_columns_indices = [2 + i * 6 for i in range(self.n_iff)]
            fy_data = data_array[:, fy_columns_indices]

            weighted_fx = np.average(fx_data, axis=0, weights=timestamps)
            weighted_fy = np.average(fy_data, axis=0, weights=timestamps)
            weighted_eff = weighted_fx * 0.9 / (Energy + 1e-7)

            self.average_Ct = weighted_fx.flatten().tolist()
            self.average_Cl = weighted_fy.flatten().tolist()
            self.average_eff = weighted_eff.flatten().tolist()
            self.average_E = Energy.flatten().tolist()
            cl_rms = np.sqrt(np.mean(fy_data**2, axis=0)).tolist()
            cl_variance = np.sum((weighted_fy-fy_data)**2, axis=0) / steps
            cl_v = cl_variance.tolist()

            full_s_list = self.average_Ct + self.average_Cl + self.average_eff + self.average_E + cl_v + cl_rms 

            with open(self.countfile, mode='a', newline='') as file:
                writer = csv.writer(file)

                # writer.writerow(self.average_Ct)
                writer.writerow(full_s_list)
            # np.savetxt(self.countname, np.array([self.n_iff, iter]))
            np.savetxt(self.countname, np.array([self.n_iff, 1]))

        print('Iter:', iter)
        # np.savetxt(self.countname, np.array([self.n_iff, iter]))

    def refresh(self, train_time, add_t):
        # No need to pub midvalue here
        # self.angles_msg.angles = self.mid_values.flatten().tolist()
        # self.pub.publish(self.angles_msg)
        time.sleep(1)

        self.tank.stop()
        time.sleep(30)
        if train_time < self.refresh_time:
            print('sleep:', self.refresh_time - train_time + add_t)
            time.sleep(self.refresh_time - train_time + add_t)

    def load_midvalue(self, midvalue):
        # For adjusting the 'zero' of each motor
        for i in range(self.n_iff):
            for j in range(self.action_dim):
                self.mid_values[i, j] = midvalue[i * self.action_dim + j]
                # self.mid_values[i, 1] = midvalue[i+10]
                # self.mid_values[i, 2] = midvalue[i+10]

        self.lower_values = self.mid_values - 60.0
        self.upper_values = self.mid_values + 60.0

        self.ini_mid_values = self.mid_values

        print('Initial Midvalue:', self.mid_values)


    def adjust_midvalue(self, increment):
        self.mid_values = self.ini_mid_values + increment
        # self.lower_values = self.mid_values - 25.0
        # self.upper_values = self.mid_values + 25.0
        self.angles_msg.angles = self.mid_values.flatten().tolist()
        self.pub.publish(self.angles_msg)
        time.sleep(5)
        



    def reset_subscriber(self):
        if self.subscriber is not None:
            self.subscriber.unregister()
        self.starting_time = time.time()
        self.subscriber = rospy.Subscriber('sensor_data', SensorMsg, self.sensor_data_callback)

    def process_track(self, filter_obj, data):
        return filter_obj.filter_data(data)


    def sensor_data_callback(self, msg):
        st = time.time()
        cumulative_time = st - self.collect_time
        msg_list = [
            msg.Ffx1, msg.Ffy1, msg.Ffz1, msg.Tfx1, msg.Tfy1, msg.Tfz1,
                       msg.Ffx2, msg.Ffy2, msg.Ffz2, msg.Tfx2, msg.Tfy2, msg.Tfz2,
                       msg.Ffx3, msg.Ffy3, msg.Ffz3, msg.Tfx3, msg.Tfy3, msg.Tfz3,
                       msg.Ffx4, msg.Ffy4, msg.Ffz4, msg.Tfx4, msg.Tfy4, msg.Tfz4,
                       msg.Ffx5, msg.Ffy5, msg.Ffz5, msg.Tfx5, msg.Tfy5, msg.Tfz5,
                       msg.Ffx6, msg.Ffy6, msg.Ffz6, msg.Tfx6, msg.Tfy6, msg.Tfz6,
                       msg.Ffx7, msg.Ffy7, msg.Ffz7, msg.Tfx7, msg.Tfy7, msg.Tfz7,
                       msg.Ffx8, msg.Ffy8, msg.Ffz8, msg.Tfx8, msg.Tfy8, msg.Tfz8
        ]
        self.current_sensor_data.append(msg_list)


        # tt = time.time()
        # shape = 48
        # ss_array = np.array(self.current_sensor_data)
        # filted_data = np.zeros(shape)
        # print('ss array shape', ss_array.shape, ' len ', len(ss_array.shape))
        
        # # if len(ss_array.shape) > 1:
        # with ThreadPoolExecutor(max_workers=9) as executor:

        #     filted = [executor.submit(self.process_track, self.massive_filters[i], ss_array[:, i]) for i in range(shape)]
        #     f_list = [filtfilt.result()[-1] for filtfilt in filted]


        f_list = [msg.Fx1, msg.Fy1, msg.Fz1, msg.Tx1, msg.Ty1, msg.Tz1,
                       msg.Fx2, msg.Fy2, msg.Fz2, msg.Tx2, msg.Ty2, msg.Tz2,
                       msg.Fx3, msg.Fy3, msg.Fz3, msg.Tx3, msg.Ty3, msg.Tz3,
                       msg.Fx4, msg.Fy4, msg.Fz4, msg.Tx4, msg.Ty4, msg.Tz4,
                       msg.Fx5, msg.Fy5, msg.Fz5, msg.Tx5, msg.Ty5, msg.Tz5,
                       msg.Fx6, msg.Fy6, msg.Fz6, msg.Tx6, msg.Ty6, msg.Tz6,
                       msg.Fx7, msg.Fy7, msg.Fz7, msg.Tx7, msg.Ty7, msg.Tz7,
                       msg.Fx8, msg.Fy8, msg.Fz8, msg.Tx8, msg.Ty8, msg.Tz8]

        # msg_list.insert(0, cumulative_time)
        msg_list1 = [cumulative_time] + msg_list + f_list
        # msg_list1 = [cumulative_time] + msg_list
        self.full_sensor_data.append(msg_list1)
        # print('Test time cost', time.time() - tt)
        # print('callbackcost:', (time.time()-st))

        # dt = time.time() - self.starting_time
        # self.starting_time += dt
        # self.time_ls.append(dt)


    def step(self, action, break_t=0.035):
        # delta angles
        # self.current_sensor_data.clear()
        self.time_ls.clear()

        # Dim of actions: N_iff * 3
        self.act_array.append(action.flatten().tolist())
        old_angles = self.stepping_angles.copy()
        new_angles = old_angles + action

        self.stepping_angles = np.clip(new_angles, self.lower_values, self.upper_values)
        self.pe = np.abs(self.stepping_angles - new_angles)

        s_action = (self.stepping_angles - old_angles) / self.action_interval
        rots = s_action * np.pi / 90.0

        # print('delay1', time.time() - time1)
        break_t = np.clip(break_t, 0.028, 0.036)
        # print('break t', break_t)
        break_t /= self.action_interval

        for i in range(self.action_interval):
            real_angle = old_angles + (i + 1) * s_action
            self.angles_msg.angles = real_angle.flatten().tolist()
            # Making it 1d list
            # print(str(self.angles_msg))
            self.pub.publish(self.angles_msg)
            time.sleep(break_t)


        if not self.current_sensor_data:
            self.running_cl.append(np.zeros(self.n_iff))

            if len(self.running_cl) > self.cl_list_len:
                self.running_cl.pop(0)

            cls = np.array(self.running_cl)

            self.fy_rms = np.sqrt(np.mean(cls ** 2, axis=0))

            pre_fys = 20 * (self.fy_rms ** 2) / 19.0

            avg_cl = np.expand_dims(np.average(cls, axis=0), axis=-1)
            zero_state = np.zeros((self.n_iff, 6))
            zero_state = np.hstack(((self.stepping_angles - self.ini_mid_values) * pi / 180.0, zero_state, rots, avg_cl))
            # done = self.is_done(zero_state)

            Fx = zero_state[:, self.action_dim]
            Fy = zero_state[:, int(1 + self.action_dim)]

            reward, done = self.compute_reward_done(zero_state, s_action)
            self.obs_array.append(zero_state.flatten().tolist())
            return zero_state, reward, done, {'Fx': Fx, 'Fy': Fy, 'Pre': pre_fys}

        #TODO: Process sensor datas
        # N_iff * 3 + N_iff * 6
        # time2 = time.time()
        # s_len = len(self.current_sensor_data)
        raw_states = np.array(self.current_sensor_data[-8:])
        # print('raw shape', raw_states.shape)
        # weighted_averages = np.average(raw_states, axis=0, weights=self.time_ls).reshape(self.n_iff, 6)
        weighted_averages = np.average(raw_states, axis=0).reshape(self.n_iff, 6)

        self.running_cl.append(weighted_averages[:, 1])

        if len(self.running_cl) > self.cl_list_len:
            self.running_cl.pop(0)

        cls = np.array(self.running_cl)
        self.fy_rms = np.sqrt(np.mean(cls ** 2, axis=0))
        pre_fys = (20 * (self.fy_rms ** 2) - weighted_averages[:, 1] ** 2) / 19.0

        avg_cl = np.expand_dims(np.average(cls, axis=0), axis=-1)
        # print('weighted average shape', weighted_averages.shape)
        state = np.hstack(((self.stepping_angles - self.ini_mid_values) * pi / 180.0, weighted_averages, rots, avg_cl))

        reward, done = self.compute_reward_done(state, s_action)

        Fx = state[:, self.action_dim]
        Fy = state[:, int(1 + self.action_dim)]

        # done = self.is_done(state)
        self.obs_array.append(state.flatten().tolist())
        # print('delay2', time.time()-time2)

        return state, reward, done, {'Fx': Fx, 'Fy': Fy, 'Pre': pre_fys}

    def reset(self):
        self.full_sensor_data = []
        self.current_sensor_data = []
        self.time_ls = []
        self.obs_array = []
        self.act_array = []
        self.running_cl = []

        self.pe_counts = np.zeros(self.n_iff)

        self.stepping_angles = self.mid_values

        self.angles_msg.angles = self.mid_values.flatten().tolist()
        self.pub.publish(self.angles_msg)

        # self.reset_subscriber()
        self.pub_zero.publish(True)
        time.sleep(1)
        # rospy.sleep(1)
        self.pub_receive.publish(True)

        #Adjusting offset -- 'theta0'
        # No need to pub midvalue here
        # self.angles_msg.angles = self.mid_values.flatten().tolist()
        # self.pub.publish(self.angles_msg)
        # time.sleep(0.1)

        self.tank.start()
        # time.sleep(self.steady_time)

        # Collect full sensor data after steady time
        self.collect_time = time.time()
        self.full_sensor_data.clear()


        if not self.current_sensor_data:
            self.running_cl.append(np.zeros(self.n_iff))
            obs = np.zeros((self.n_iff, self.obs_dim))
        else:
            # last_elements = np.array(self.current_sensor_data[-10:])
            last_elements = np.array(self.current_sensor_data)
            obs = np.average(last_elements, axis=0).reshape(self.n_iff, 6)
            cl_running = obs[:, 1]
            self.running_cl.append(cl_running)
            rots = np.zeros((self.n_iff, self.action_dim))

            obs = np.hstack(((self.stepping_angles - self.ini_mid_values) * pi / 180.0, obs, rots, np.expand_dims(cl_running, axis=-1)))

        self.obs_array.append(obs.flatten().tolist())

        return obs

    def compute_reward_done(self, state, action):
        Fx = state[:, self.action_dim]
        Fy = state[:, int(1 + self.action_dim)]
        Mz = state[:, int(6 + self.action_dim)]
        theta = action[:, 0]
        # Clipped_Fx = np.clip(Fx, -1.0, 2.0)
        # Clipped_Fy = np.clip(Fy, -1.2, 2.0)
        Clipped_Fx = Fx
        Clipped_Fy = Fy
        # reward = np.zeros(self.n_iff)
        exps = np.exp(self.pe / 5.0) - 1.0  # -e^0
        penalty = np.sum(exps, axis=1)

        count_positive = np.sum(self.pe > 1e-3, axis=1)
        #==3 too slight penalty
        done_condition = np.array((count_positive > 1))

        # done condition can only be 0 or 1, if current condi==0 -> 
        # cummulative pe_counts set zero
        self.pe_counts += done_condition
        self.pe_counts *= done_condition

        penalty *= np.array(self.pe_counts > 8)
        penalty += np.array(self.pe_counts > 12) * penalty
        penalty += np.array(self.pe_counts > 24) * penalty

        done = self.pe_counts > 48

        penalty += np.array(done) * 11.4514 * 2

        # for i in range(self.n_iff):
        #     if penalty[i] > 1e-6:
        #         self.pe_counts[i] += 1
        #     else:
        #         self.pe_counts[i] = 0
            
        #     if self.pe_counts[i] > 9:
        #         penalty[i] += 100

        # print(penalty)
        reward = 1.5 * Clipped_Fx - 0.25 * Mz * theta - 0.05 * penalty
        # reward = 0.9 * Clipped_Fx + 0.5 * Clipped_Fy - 0.08 * Mz * theta - 0.05 * penalty
        return reward * 1.0, done

    def is_done(self, state):
        done = [False] * self.n_iff
        for i in range(self.n_iff):
            if self.pe_counts[i] > 9:
                done[i] = True
        return done

    def save_data(self):
        file = open(self.filename, mode='wb')
        pickle.dump(self.obs_array, file)
        file.close()

        # with open(self.csv_filename, mode='w') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(self.obs_array)


class ServoController:
    def __init__(self):
        self.angle1_accumulator = 0.0
        self.angle2_accumulator = 0.0
        self.old1 = 0.0
        self.old2 = 0.0

    def reset(self):
        self.angle1_accumulator = 0.0
        self.angle2_accumulator = 0.0
        self.old1 = 0.0
        self.old2 = 0.0

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

if __name__ == '__main__':
    SC = ServoController()
    SC.reset()
    env = ServoControlEnv()
    env.reset()
    try:
        while not rospy.is_shutdown():
        # while not done:
            action = SC.servo_control()
            print(action)
            env.step(action)
            env.rate.sleep()
    finally:
        print(1)
        # env.save_data()
