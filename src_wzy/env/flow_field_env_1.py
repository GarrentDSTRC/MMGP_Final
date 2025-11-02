from xmlrpc.client import ServerProxy
import subprocess
import json
import time
import random
import numpy as np
import argparse
from gym.spaces import Box
import os
import signal

def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None):
        self.observation_dim = 24
        self.action_dim = 2
        self.step_counter = 0
        self.state = [1, 2, 3, 4]
        self.action_interval = config.action_interval
        # for gym vector env
        self.unwrapped = self
        self.unwrapped.spec = None
        self.observation_space = Box(low=-1e6, high=1e6, shape=[self.observation_dim])
        self.action_space = Box(low=-2, high=2, shape=[self.action_dim])
        self.local_port = local_port

        # TODO:start the server
        while True:
            port = random.randint(6000, 8000)
            if not is_port_in_use(port):
                break
        port = port if network_port == None else network_port
        # print("Foil_env start!")
        if local_port == None:
            self.server = subprocess.Popen(f'xvfb-run -a /mnt/nasdata/runji/LilyPad/processing-4.0b6/processing-java --sketch=/mnt/nasdata/runji/LilyPad/RLnonparametric_Foil1 --run {port} {info}', shell=True)
            # wait the server to start
            time.sleep(20)
            self.proxy = ServerProxy(f"http://localhost:{port}/")
        else:
            self.proxy = ServerProxy(f"http://localhost:{local_port}/")
        # communication by following function
        ## proxy.connect.query_state()
        ## proxy.connect.reset()
        ## proxy.connect.Step({"y":1;"theta":2})

    def step(self, action):
        self.step_counter += 1
        # [alpha y CT CS]
        # TODO: need load from true environment
        # if action[0] > 1.2:
        #     action[0] = 1.2
        # elif action[0]< -1.2:
        #     action[0] = -1.2
        # if action[1] > 1.5:
        #     action[1] = 1.5
        # elif action[1]< -1.5:
        #     action[1] = -1.5
        #self.action_interval = 10
        result_ls, ct_ls, eta_ls, cp_ls, fx_ls, dt_ls = [], [], [], [], [], []
        for i in range(self.action_interval):
            step_y = float(action[0])#/self.action_interval # since velosity
            step_alpha = float(action[1])#/self.action_interval
            action_json = {"yvel": step_y, "thetavel": step_alpha}
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            state, reward, done, info = self.parseStep(res_str)
            self.reward, self.state, self.done = np.array(reward, dtype=np.float32), np.array(state, np.float32), np.array(done, np.float32)
            result_ls.append(self.reward)
            ct_ls.append(info['ct'])
            eta_ls.append(info['eta'])
            cp_ls.append(info['cp'])
            fx_ls.append(info['fx'])
            dt_ls.append(info['dt'])
            if self.done == True:
                break
        self.reward = np.average(result_ls, weights=dt_ls) ## TODO: mean?
        #print(res)
        info = {"ct": np.average(ct_ls, weights=dt_ls),
                "eta": np.average(eta_ls, weights=dt_ls),
                "cp": np.average(cp_ls, weights=dt_ls),
                "fx": np.average(fx_ls, weights=dt_ls),
                "dt": np.sum(dt_ls),
                "delta_y": info['delta_y'],
                "delta_theta": info['delta_theta']}


        return self.state, self.reward, self.done, info

    def reset(self):
        # TODO: reset true environment
        self.proxy.connect.reset()
        action_json = {"yvel": 0, "thetavel": 0}
        res_str = self.proxy.connect.Step(json.dumps(action_json))
        state, reward, done, info = self.parseStep(res_str)
        self.reward, self.state, self.done = np.array(reward, dtype=np.float32), np.array(state, np.float32), np.array(done, np.float32)
        return self.state

    def reset_sin(self):
        # TODO: reset true environment
        self.proxy.connect.reset()
        action_json = {"yvel": 0, "thetavel": 0}
        res_str = self.proxy.connect.Step(json.dumps(action_json))
        state, reward, done, info = self.parseStep(res_str)
        self.reward, self.state, self.done = np.array(reward, dtype=np.float32), np.array(state, np.float32), np.array(done, np.float32)
        return self.state, info


    def parseStep(self, info):
        all_info = json.loads(info)
        state = json.loads(all_info['state'][0])
        reward = all_info['reward']
        done = all_info['done']
        state['SparsePressure'] = list(map(float, state['SparsePressure'].split('_')))
        # TODO: normarlization state
        state_ls = [state['delta_y']/30, state['delta_theta']/50, state['x_velocity']/30, state['theta_velocity']/30] + list(np.tanh(np.array(state['SparsePressure'])/20))#state['eta']
        state_ls = list(np.nan_to_num(np.array(state_ls), nan=0))
        if not done:
            # if not done, we cal ct and eta as reward
            # TODO: rescale the two-dimension reward
            ct = np.clip(state['ct'], -30, 30)/10
            eta = np.clip(state['eta'], -3, 3)
            cp = np.clip(state['cp'], -300, 300)/100
            dt = state['dt']
            #reward = 1

            # if shift far from the center, we give a penalty
            if np.abs(state['delta_y']) > 60:
                reward -= (np.abs(state['delta_y'])-60)**2/100

            reward += ct -cp +1#+ ct #TODO: add dt?
        if done and reward == -300:
             reward = -300

        return state_ls, reward, done, {'ct': state['ct'], 'eta': state['eta'], 'fx': -16*state['ct'], 'cp': state['cp'], 'dt': state['dt'], 'delta_y': state['CurrentY'], 'delta_theta': state['CurrentPhi']}

    def parseState(self, state):
        state = json.loads(json.loads(state)['state'][0])
        state['SparsePressure'] = list(map(float, state['SparsePressure'].split('_')))

        ## TODO: define different combination for state
        state_ls = [state['delta_y'], state['y_velocity'], state['eta'], state['delta_theta'], state['x_velocity'],
                    state['theta_velocity']] + state['SparsePressure']
        state_ls = list(np.nan_to_num(np.array(state_ls), nan=0))
        return state_ls

    def terminate(self):
        pid = os.getpgid(self.server.pid)
        self.server.terminate()
        os.killpg(pid, signal.SIGTERM)  # Send the signal to all the process groups.

    def close(self):
        if self.local_port == None:
            self.server.terminate()

class fake_env:
    def __init__(self, config):
        self.observation_dim = 0
        self.action_dim = 2
        self.step_counter = 0
        self.state = [1, 2, 3, 4]
        # TODO: need build client connection


    def step(self, action):
        self.step_counter += 1
        # [alpha y CT CS]
        # TODO: need load from true environment
        self.state =  [1, 2, 3, 4]
        self.reward = - (0.5 - action[0]) **2 - (0.5 - action[1]) **2 # action is two-dimensional: y, alpha
        self.done = True if self.step_counter >= 100 else False

        return self.state, self.reward, self.done

    def reset(self):
        # TODO: reset true environment
        self.step_counter = 0
        return self.state



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.action_interval = 1
    env = foil_env(args)
    env.reset()
    done = False
    while done != True:
        state, reward, done, _ = env.step([0.27, 0.97])
        print(done)

    env.step([0, 0])
    env.reset()

    #myserver.start_server()
