from gym.spaces import Box
import os
import signal
import rospy
import gym
from gym import spaces
import numpy as np
from ss.msg import SensorMsg, MotorAngles
import csv
import os
from datetime import datetime
import math
import time


class ServoControlEnv(gym.Env):
    def __init__(self):
        super(ServoControlEnv, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.n_iff = 2

        home_directory = os.path.expanduser("~")
        self.csv_filename = os.path.join(home_directory, "sensor_data.csv")

        self.data_array = []

        rospy.init_node('controller', anonymous=True)

        self.pub = rospy.Publisher('motor_control_all', MotorAngles, queue_size=10)
        self.subscriber = rospy.Subscriber('sensor_data', SensorMsg, self.sensor_data_callback)
        #TODO: Check where to put the subscriber
        self.rate = rospy.Rate(20)

        self.angles_msg = MotorAngles()
        self.current_sensor_data = []
        self.time_ls = []
        self.starting_time = time.time()
        self.action_interval = 10
        self.stepping_angles = np.ones([self.n_iff, 3]) * 180.0

    def start(self):
        rospy.spin()

    def reset_subscriber(self):
        if self.subscriber is not None:
            self.subscriber.unregister()
        self.starting_time = time.time()
        self.subscriber = rospy.Subscriber('sensor_data', SensorMsg, self.sensor_data_callback)

    def sensor_data_callback(self, msg):
        self.current_sensor_data.append([msg.Fx, msg.Fy, msg.Fz, msg.Tx, msg.Ty, msg.Tz,
                       msg.Fx2, msg.Fy2, msg.Fz2, msg.Tx2, msg.Ty2, msg.Tz2])

        dt = time.time() - self.starting_time
        self.starting_time += dt
        self.time_ls.append(dt)


    def step(self, action):
        # delta angles
        time1 = time.time()
        self.current_sensor_data.clear()
        self.time_ls.clear()
        # angle1 = 180 + 30 * action[:, 0]
        # angle2 = 180 + 30 * action[:, 1]
        # angle3 = 180 + 30 * action[:, 2]
        #
        # #TODO: clip maximum angles
        # angle1 = np.max(120, np.min(angle1, 240))
        # angle2 = np.max(120, np.min(angle2, 240))
        # angle3 = np.max(120, np.min(angle3, 240))
        # s_action = action / self.action_interval

        # Dim of actions: N_iff * 3
        old_angles = self.stepping_angles.copy()
        self.stepping_angles = np.clip(self.stepping_angles + action, 120, 240)
        s_action = (self.stepping_angles - old_angles) / self.action_interval
        print('delay1', time.time() - time1)

        for i in range(self.action_interval):
            real_angle = old_angles + (i + 1) * s_action
            self.angles_msg.angles = real_angle.flatten().tolist()
            #Making it 1d list
            self.pub.publish(self.angles_msg)
            time.sleep(0.004)


        # time.sleep(0.05)


        if not self.current_sensor_data:
            return np.zeros(14), 0, False, {}

        #TODO: Process sensor datas
        # N_iff * 3 + N_iff * 6
        raw_states = np.array(self.current_sensor_data)
        weighted_averages = np.average(raw_states, axis=1, weights=self.time_ls).reshape(self.n_iff, 6)
        state = np.hstack((self.stepping_angles, weighted_averages))
        # state = np.array([
        #     angle1, angle2, angle3,
        #     self.current_sensor_data.Fx, self.current_sensor_data.Fy, self.current_sensor_data.Fz,
        #     self.current_sensor_data.Tx, self.current_sensor_data.Ty, self.current_sensor_data.Tz,
        #     self.current_sensor_data.Fx2, self.current_sensor_data.Fy2, self.current_sensor_data.Fz2,
        #     self.current_sensor_data.Tx2, self.current_sensor_data.Ty2, self.current_sensor_data.Tz2
        # ])

        reward = self.compute_reward(state)

        done = self.is_done(state)
        self.data_array.append(state)

        return state, reward, done, {}

    def reset(self):
        time.sleep(0.1)

        self.current_sensor_data = []
        self.time_ls = []

        self.current_sensor_data = None
        self.reset_subscriber()
        return np.zeros(14)

    def compute_reward(self, state):
        reward = 0
        return reward

    def is_done(self, state):
        done = False
        return done

    def save_data(self):
        with open(self.csv_filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerows(self.data_array)


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

        self.angle1_accumulator += 20
        self.angle2_accumulator += 20

        # 计算新的角度值
        theta1 = 30 * math.sin(math.radians(self.angle1_accumulator))
        theta2 = (30 + 25) * math.sin(math.radians(self.angle2_accumulator))

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
            action = SC.servo_control()
            env.step(action)
        env.rate.sleep()
    finally:
        env.save_data()
