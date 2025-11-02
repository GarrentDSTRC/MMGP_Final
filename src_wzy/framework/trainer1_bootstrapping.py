import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical
import torch.nn as nn
import os
from einops import rearrange, repeat
from torch.optim import Adam
# from framework.buffer1 import Replay_buffer
from model.actor1 import GaussianActor, RActor
from model.critic1 import Critic, RCritic
import wandb
from model.mae import TransformerAgent
from model.mae_variable_sigma import TransformerAgentVS
from model.mlp import MLPAgent
from framework.buffer import Buffer as buffer
from datetime import datetime



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(7)

def trajectory_property():
    return ["action", "hidden", "next_hidden", "hidden_q",
                                 "next_hidden_q", "hidden_q_target",
                                 "next_hidden_q_target", "id"]
    # return ["action", "id"]

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for step_idx in reversed(range(td_delta.shape[-1])):
        delta = td_delta[:, step_idx]
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float).T

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def update_params(optim, loss, clip=False, param_list=False, retain_graph=False):
    optim.zero_grad()
    loss.backward()
    if clip is not False:
        for i in param_list:
            nn.utils.clip_grad_norm_(i, clip)
    optim.step()

class TPPO_B(object):
    def __init__(self, args):
        # env parameters
        self.state_dim = args.obs_space
        self.action_dim = args.action_space
        self.env_num = args.env_num
        self.device = args.device

        # model parameters
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        self.max_seq_len = args.context_len

        # training parameters
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lamda = args.gae_lambda
        self.clip = args.ppo_clip
        self.epoch = args.ppo_epoch
        self.entropy = args.ppo_entropy
        self.grad_norm = args.grad_norm_clip
        self.actor_loss = 0
        self.critic_loss = 0
        self.boots_num = args.boots_num

        # define actor and critic
        args.mode = "actor"
        self.actors = []
        self.critics = []
        if args.mlp == True:
            self.actor = MLPAgent(args).to(self.device)
        else:
            if args.sigma_type == 'share':
                for i in range(4):
                    actor = TransformerAgent(args).to(self.device)
                    self.actors.append(actor)
                # for i in range(2):
                #     actor = TransformerAgentVS(args).to(self.device)
                #     self.actors.append(actor)

            else:
                self.actor = TransformerAgentVS(args).to(self.device)

        
        
        
        args.mode = "critic"
        if args.mlp == True:
            self.critic = MLPAgent(args).to(self.device)
        else:
            for i in range(4):
                critic = TransformerAgent(args).to(self.device)
                self.critics.append(critic)
            # self.critic = TransformerAgent(args).to(self.device)

        # define for transformer
        self.state = torch.zeros(self.env_num, self.max_seq_len ,self.state_dim).to(self.device)

        # define buffer
        self.memory = buffer(args)


    def reset_optimizer(self):
        for actor in self.actors:
            actor.reset_optimizer()
        self.critic.reset_optimizer()

    def reset_state(self):
        self.state = torch.zeros(self.env_num, self.max_seq_len ,self.state_dim).to(self.device)

    def choose_action(self, state, id, train=True):
        # TODO: [0807]inference by context
        state = torch.FloatTensor(state).unsqueeze(1).to(self.device)
        self.state = torch.cat([self.state, state], dim=1)[:, 1:, :]
        if train:
            action, _ = self.actors[id].getVecAction(self.state)
        else:
            action, _ = self.actors[id].getVecAction(self.state, train=False)
        return action

    def learn(self, id_list, ep_list, single=False):
        # input dim [batch * context_len * dim], batch -> [num_envs * time_steps]
        state_np, action_np, reward_np, done_np, next_state_np = self.memory.sample(self.batch_size)

        number = len(id_list)
        starting_index = 0
        ending_index = 0

        for iter in range(number):
            id = id_list[iter]
            ending_index += ep_list[iter]

            state = torch.FloatTensor(state_np[:, starting_index:ending_index, :, :]).to(self.device)
            action = torch.FloatTensor(action_np[:, starting_index:ending_index, :]).to(self.device)
            reward = torch.FloatTensor(reward_np[:, starting_index:ending_index]).to(self.device)
            next_state = torch.FloatTensor(next_state_np[:, starting_index:ending_index, :, :]).to(self.device)
            done = torch.FloatTensor(done_np[:, starting_index:ending_index]).to(self.device)
            mu_action_log_prob = self.actors[id].getActionLogProb(state, action).detach()

            td_target = reward + self.gamma * self.critic.getValue(next_state).reshape(state.shape[0],
                                                                                       state.shape[1]) * (
                                    1 - done)  # 20*1
            td_error = td_target - self.critic.getValue(state).reshape(state.shape[0], state.shape[1])
            advantage = compute_advantage(self.gamma, self.lamda, td_error.cpu()).to(self.device)
            # [trick] : advantage normalization
            td_lamda_target = advantage + self.critic.getValue(state).reshape(state.shape[0], state.shape[1])
            advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5)).detach()
            old_action_log_prob = []

            for actor in self.actors:
                old_action_log_prob.append(actor.getActionLogProb(state, action))

            for _ in range(self.epoch):
                # sample new action and new action log prob
                print(_)

                for index in range(self.boots_num):
                    new_action_log_prob = self.actors[index].getActionLogProb(state, action, train=True)
                    ratio = (new_action_log_prob - mu_action_log_prob).exp()
                    surprise = ratio * advantage
                    ratio_old = (old_action_log_prob[index] - mu_action_log_prob).exp()
                    clipped_surprise = torch.clamp(ratio, (1 - self.clip) * ratio_old, (1 + self.clip) * ratio_old) * advantage
                    actor_loss = -torch.min(surprise, clipped_surprise).mean()

                    self.actors[index].optimizer.zero_grad()
                    actor_loss.backward()

                    nn.utils.clip_grad_norm_(self.actors[index].parameters(), self.grad_norm)
                    self.actors[index].optimizer.step()

                    self.clipfrac = torch.mean(torch.greater(torch.abs(ratio - 1), self.clip).float()).item()
                    self.approxkl = torch.mean(-new_action_log_prob + mu_action_log_prob).item()

                    self.actor_loss = actor_loss.item()

                # update critic
                critic_loss = F.mse_loss(td_lamda_target.detach(),
                                         self.critic.getValue(state).reshape(state.shape[0], state.shape[1]))
                # update

                self.critic.optimizer.zero_grad()

                critic_loss.backward()
                # trick: clip gradient

                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                self.critic.optimizer.step()
                self.critic_loss = critic_loss.item()


        return self.actor_loss, self.critic_loss


    def learn_vtrace(self, id_list, ep_list, single=False):
    # input dim [batch * context_len * dim], batch -> [num_envs * time_steps]
        state_np, action_np, reward_np, done_np, next_state_np = self.memory.sample(self.batch_size)

        number = len(id_list)
        starting_index = 0
        ending_index = 0

        for iter in range(number):
            id = id_list[iter]
            ending_index += ep_list[iter]

            state = torch.FloatTensor(state_np[:, starting_index:ending_index, :, :]).to(self.device)
            action = torch.FloatTensor(action_np[:, starting_index:ending_index, :]).to(self.device)
            reward = torch.FloatTensor(reward_np[:, starting_index:ending_index]).to(self.device)
            next_state = torch.FloatTensor(next_state_np[:, starting_index:ending_index, :, :]).to(self.device)
            done = torch.FloatTensor(done_np[:, starting_index:ending_index]).to(self.device)
            mu_action_log_prob = self.actors[id].getActionLogProb(state, action).detach()

            IS_ls = []
            delta_ls = []
            vtrace_ls = []
            for index in range(self.boots_num):
                old_logprob = self.actors[index].getActionLogProb(state, action)

                IS = (old_logprob - mu_action_log_prob).exp()
                IS_ls.append(IS)

                c = torch.min(torch.tensor(1.0), IS)

                v_next = self.critics[index].getValue(next_state).reshape(state.shape[0], state.shape[1])
                v = self.critics[index].getValue(state).reshape(state.shape[0], state.shape[1])
                td_target = reward + self.gamma * v_next * (1 - done)  # 20*1
                td_error = td_target - v


                # V-trace correction for Advantage Estimation
                v_trace_target = td_target.clone()
                delta = td_error.clone()

                dlst = v[:, -1] + td_target[:, -1] * c[:, -1] + self.gamma * c[:, -1] * td_error[:, -1]

                v_trace_target[:, -1] = dlst
                delta[:, -1] = reward[:, -1] + self.gamma * dlst - v[:, -1]

                # Use V-trace update
                for t in reversed(range(state.shape[0] - 1)):
                    v_trace_target[t] = v[t] + td_error[t] * c[t] + self.gamma * c[t] * (v_trace_target[t+1] - v_next[t+1])
                    delta[t] = reward[t] + self.gamma * v_trace_target[t+1] - v[t]

                    # v_trace_target[t] = reward[t] + self.gamma * v_next[t + 1] * (1 - done[t]) + self.lambda_ * delta[
                    #     t + 1]
                    # delta[t] = v_trace_target[t] - self.actors[id].critic.getValue(state[t]).reshape(-1)

                # Normalization of Advantage for stable training
                # v_trace_target = (v_trace_target - v_trace_target.mean()) / (v_trace_target.std() + 1e-5)
                # delta = (delta - delta.mean()) / (delta.std() + 1e-6)
                delta_ls.append(delta.detach())
                vtrace_ls.append(v_trace_target.detach())


            for _ in range(self.epoch):
                for index in range(self.boots_num):
                    new_action_log_prob = self.actors[index].getActionLogProb(state, action, train=True)
                    ratio = (new_action_log_prob - mu_action_log_prob).exp()
                    delta = delta_ls[index]
                    surprise = ratio * delta  # Use delta instead of GAE-based advantage
                    # ratio_old = (old_action_log_prob[index] - mu_action_log_prob).exp()
                    ratio_old = IS_ls[index]
                    v_trace_value = vtrace_ls[index]
                    
                    clipped_surprise = torch.clamp(ratio, (1 - self.clip) * ratio_old, (1 + self.clip) * ratio_old) * delta
                    actor_loss = -torch.min(surprise, clipped_surprise).mean()


                    self.actors[index].optimizer.zero_grad()
                    actor_loss.backward()

                    nn.utils.clip_grad_norm_(self.actors[index].parameters(), self.grad_norm)
                    self.actors[index].optimizer.step()

                    critic_loss = F.mse_loss(v_trace_value.detach(),
                                             self.critics[id].getValue(state).reshape(state.shape[0],
                                                                                            state.shape[1]))
                    self.critics[index].optimizer.zero_grad()
                    critic_loss.backward()

                    nn.utils.clip_grad_norm_(self.critics[index].parameters(), self.grad_norm)

                    self.critics[index].optimizer.step()




                    self.clipfrac = torch.mean(torch.greater(torch.abs(ratio - 1), self.clip).float()).item()
                    self.approxkl = torch.mean(-new_action_log_prob + mu_action_log_prob).item()

                    self.actor_loss = actor_loss.item()
                    self.critic_loss = critic_loss.item()

                # # Update critic using V-trace target
                # critic_loss = F.mse_loss(v_trace_target.detach(),
                #                         self.actors[id].critic.getValue(state).reshape(state.shape[0], state.shape[1]))
                #
                # self.actors[id].critic.optimizer.zero_grad()
                # critic_loss.backward()
                # nn.utils.clip_grad_norm_(self.actors[id].critic.parameters(), self.grad_norm)
                # self.actors[id].critic.optimizer.step()
                #
                # self.critic_loss = critic_loss.item()

        return self.actor_loss, self.critic_loss

    
    def learn_critic(self):
        # input dim [batch * context_len * dim], batch -> [num_envs * time_steps]
        state, action, reward, done, next_state = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        ## ppo update
        td_target = reward + self.gamma * self.critic.getValue(next_state).reshape(state.shape[0], state.shape[1]) * (1 - done) #20*1
        td_error = td_target - self.critic.getValue(state).reshape(state.shape[0], state.shape[1])
        advantage = compute_advantage(self.gamma, self.lamda, td_error.cpu()).to(self.device)
        # [trick] : advantage normalization
        td_lamda_target = advantage + self.critic.getValue(state).reshape(state.shape[0], state.shape[1])



        for _ in range(self.epoch-2):
            # sample new action and new action log prob
            print(_)

            # update critic
            critic_loss = F.mse_loss(td_lamda_target.detach(), self.critic.getValue(state).reshape(state.shape[0], state.shape[1]))
            # update

            self.critic.optimizer.zero_grad()

            critic_loss.backward()
            # trick: clip gradient

            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)

            self.critic.optimizer.step()

        # the fraction of the training data that triggered the clipped objective

        self.critic_loss = critic_loss.item()

        return critic_loss

    def insert_data(self, data):
        for k, v in data.items():
            self.memory.insert(k, v)

    def imitation_train(self, offline_data, iter=7000):
        action_set = offline_data[0]
        state_set = offline_data[1]
        # state_ls = [state_set['y'] / 30, state_set['theta'] / 50, state_set['yvel'] / 30, state_set['tvel'] / 30]
        # concatenate state list
        # state_ls = np.vstack(state_ls).T
        # convert to tensor
        state_ls = torch.tensor(state_set, dtype=torch.float).to(self.device)
        action = torch.tensor(action_set, dtype=torch.float).to(self.device)

        for iter in range(iter):
            self.policy.set_h()
            policy_action_ls = []
            for i in range(len(action_set)):
                state = state_ls[i].view(1, 1, -1).to(self.device)
                next_action, next_action_logprob, _, _, _ = self.policy.sample(state)
                policy_action_ls.append(next_action)
            # concatenate policy action list
            policy_action = torch.cat(policy_action_ls, dim=0).squeeze()
            # cal the loss for action and policy_action
            loss = F.mse_loss(policy_action, action)
            self.imit_loss = loss.detach().cpu().numpy()
            self.policy_optim.zero_grad()
            loss.backward()
            self.policy_optim.step()
            if iter % 1 == 0:
                print("imitation train iter: ", iter, " loss: ", self.imit_loss)
                wandb.log({"imitation_train_loss": self.imit_loss})

        return loss.detach().cpu().numpy()


    def save_critic(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)

    def save(self, save_path, episode):
        time = datetime.now()
        time_str = time.strftime('%m_%d_%H_%M')
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        # model_critic_path = os.path.join(base_path, "critic_" + str(episode) + '_' + time_str + ".pth")
        # torch.save(self.critic.state_dict(), model_critic_path)

        for ids in range(self.boots_num):
            model_actor_path = os.path.join(base_path, "actor_" + str(ids) + '_' + str(episode) + '_' + time_str + ".pth")
            torch.save(self.actors[ids].state_dict(), model_actor_path)
            model_c_path = os.path.join(base_path, "critic_" + str(ids) + '_' + str(episode) + '_' + time_str + ".pth")
            torch.save(self.critics[ids].state_dict(), model_c_path)



    def load(self, file):
        # load data from file, and map to the correct device
        # self.actor.load_state_dict(torch.load(file, map_location='cuda:0'))
        self.actor.load_state_dict(torch.load(file,  map_location=self.device), strict=False)
        self.actor.reset_optimizer()

    def bootstrap_load(self, files):
        # idd = 0
        for idd in range(4):
            print(idd)
            self.actors[idd].load_state_dict(torch.load(files[idd], map_location=self.device), strict=False)
            self.actors[idd].reset_optimizer()
 

    def load_v(self, file):
        # load data from file, and map to the correct device
        # self.actor.load_state_dict(torch.load(file, map_location='cuda:0'))
        for idd in range(4):
            self.critics[idd].load_state_dict(torch.load(file, map_location=self.device), strict=False)
            self.critics[idd].reset_optimizer()

class RSAC(object):
    def __init__(self, args):
        self.state_dim = args.obs_space
        self.action_dim = args.action_space
        self.hidden_size = args.hidden_size
        self.hid_layer = args.num_hid_layer
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.alpha = args.alpha
        self.tune_entropy = args.tune_entropy
        self.target_entropy_ratio = args.target_entropy_ratio
        self.buffer_size = args.buffer_capacity
        self.batch_size = args.batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy = RActor(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_size,
            action_dim=self.action_dim,
            tanh=False,
            action_low=-100,
            action_high=100
        ).to(self.device)
        self.policy.apply(weights_init_)
        self.q1 = RCritic(self.state_dim+self.action_dim, 1, self.hidden_size, self.hid_layer).to(self.device)
        self.q1.apply(weights_init_)
        self.q1_target = RCritic(self.state_dim+self.action_dim, 1, self.hidden_size, self.hid_layer).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.critic_optim = Adam(self.q1.parameters(), lr=self.critic_lr)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.critic_lr)
        self.memory = buffer(self.buffer_size, trajectory_property())
        self.memory.init_item_buffers()
        self.a_loss = 0
        self.c_loss = 0
        self.alpha_loss = 0
        self.gamma = args.gamma
        self.learn_step_counter = 0
        self.ent = 0
        self.target_replace_iter = args.target_replace
        self.tau = args.tau

    def reset_optim(self):
        self.critic_optim = Adam(self.q1.parameters(), lr=self.critic_lr)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.critic_lr)


    def choose_action(self, state, train=True):
        obs = torch.tensor(state, dtype=torch.float).view(1, 1, -1).to(self.device)
        if train:
            action, _, _, hidden, next_hidden = self.policy.sample(obs)
            # action = action.detach()
            action = action.detach().squeeze(0).cpu().numpy()
            t_action = torch.tensor(action, dtype=torch.float).view(1, 1, -1).to(self.device)
            ## inference Q for hidden state
            self.q1(torch.cat([obs, t_action], -1))
            self.q1_target(torch.cat([obs, t_action], -1))
            hidden_q, next_hidden_q, hidden_q_target, next_hidden_q_target = self.q1.hidden_q.detach().squeeze(
                0).cpu().numpy(), self.q1.next_hidden_q.detach().squeeze(
                0).cpu().numpy(), self.q1_target.hidden_q.detach().squeeze(
                0).cpu().numpy(), self.q1_target.next_hidden_q.detach().squeeze(0).cpu().numpy()

            self.add_experience({"action": action, "hidden": hidden, "next_hidden": next_hidden, "hidden_q": hidden_q,
                                 "next_hidden_q": next_hidden_q, "hidden_q_target": hidden_q_target,
                                 "next_hidden_q_target": next_hidden_q_target})

            # self.add_experience({"action": action})
        else:
            _, _, action, _, _ = self.policy.sample(obs)
            action = action.detach()
        return {'action': action}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def reset_rnn(self, h=None):
        self.policy.set_h()
        self.q1.set_h()
        self.q1_target.set_h()

    def learn(self):
        data = self.memory.sample2(self.batch_size)

        obss = torch.tensor(data["states"], dtype=torch.float).squeeze().to(self.device)
        obs_s = torch.tensor(data["states_next"], dtype=torch.float).squeeze().to(self.device)
        rewards = torch.tensor(data["rewards"], dtype=torch.float).squeeze().to(self.device)
        dones = torch.tensor(data["dones"], dtype=torch.float).squeeze().to(self.device)
        actions = torch.tensor(data["action"], dtype=torch.float).squeeze().to(self.device)
        hiddens = torch.tensor(data["hidden"], dtype=torch.float).squeeze().to(self.device) # timlens * batch *dim
        hidden_qs = torch.tensor(data["hidden_q"], dtype=torch.float).squeeze().to(self.device)
        hidden_q_targets = torch.tensor(data["hidden_q_target"], dtype=torch.float).squeeze().to(self.device)
        ids = torch.tensor(data["id"], dtype=torch.float).squeeze().to(self.device)

        policy_hidden = self.policy.h
        q1_hidden = self.q1.h
        q1_target_hidden = self.q1_target.h

        sip = np.where(data["id"][0] == 0)[0]
        sip = sip[1:] if sip[0] == 0 else sip
        obss = np.split(obss, sip)
        obs_s = np.split(obs_s, sip)
        rewards = np.split(rewards, sip)
        dones = np.split(dones, sip)
        actions = np.split(actions, sip)
        hiddens = np.split(hiddens, sip)
        hidden_qs = np.split(hidden_qs, sip)
        hidden_q_targets = np.split(hidden_q_targets, sip)
        id = np.split(ids, sip)
        for obs, obs_, reward, done, action, hidden, hidden_q, hidden_q_target in zip(obss, obs_s, rewards, dones
                , actions, hiddens, hidden_qs, hidden_q_targets):
            obs = obs.unsqueeze(0)
            obs_ = obs_.unsqueeze(0)
            self.policy.set_h(hidden[0].view(1, 1, -1))
            self.q1.set_h(hidden_q[0].view(1, 1, -1))
            self.q1_target.set_h(hidden_q_target[0].view(1, 1, -1))
            with torch.no_grad():
                next_action, next_action_logprob, _, _, _ = self.policy.sample(obs_)
                target_next_q = self.q1_target(
                    torch.cat([obs_, next_action], -1)).squeeze() - (self.alpha * next_action_logprob).squeeze()
                next_q_value = reward.squeeze() + (1 - done.squeeze()) * self.gamma * target_next_q
            qf1 = self.q1(torch.cat([obs, action.unsqueeze(0)], -1)).squeeze()  # TD-lambda
            qf_loss = F.mse_loss(qf1, next_q_value)
            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()
            self.policy.set_h(hidden[0].view(1, 1, -1))
            self.q1.set_h(hidden_q[0].view(1, 1, -1))
            self.q1_target.set_h(hidden_q_target[0].view(1, 1, -1))
            pi, log_pi, _, _, _ = self.policy.sample(obs)
            qf_pi = self.q1(torch.cat([obs, pi], -1))
            policy_loss = ((self.alpha * log_pi) - qf_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
            self.c_loss = qf_loss.detach().cpu().numpy()
            self.a_loss = policy_loss.detach().cpu().numpy()
            entropies = -torch.sum(log_pi.exp() * log_pi, dim=1, keepdim=True)  # [batch, 1]
            entropies = entropies.detach().cpu().numpy().mean()
            self.ent = entropies
            if self.learn_step_counter % self.target_replace_iter == 0:
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)
                # for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                #    target_param.data.copy_(self.tau * param.data + (1.-self.tau) * target_param.data)

            # recover the hidden
            self.policy.set_h(policy_hidden)
            self.q1.set_h(q1_hidden)
            self.q1_target.set_h(q1_target_hidden)

    def imitation_train(self, offline_data, iter=7000):
        action_set = offline_data[0]
        state_set = offline_data[1]
        # state_ls = [state_set['y'] / 30, state_set['theta'] / 50, state_set['yvel'] / 30, state_set['tvel'] / 30]
        # concatenate state list
        # state_ls = np.vstack(state_ls).T
        # convert to tensor
        state_ls = torch.tensor(state_set, dtype=torch.float).to(self.device)
        action = torch.tensor(action_set, dtype=torch.float).to(self.device)

        for iter in range(iter):
            self.policy.set_h()
            policy_action_ls = []
            for i in range(len(action_set)):
                state = state_ls[i].view(1, 1, -1).to(self.device)
                next_action, next_action_logprob, _, _, _ = self.policy.sample(state)
                policy_action_ls.append(next_action)
            # concatenate policy action list
            policy_action = torch.cat(policy_action_ls, dim=0).squeeze()
            # cal the loss for action and policy_action
            loss = F.mse_loss(policy_action, action)
            self.imit_loss = loss.detach().cpu().numpy()
            self.policy_optim.zero_grad()
            loss.backward()
            self.policy_optim.step()
            if iter % 1 == 0:
                print("imitation train iter: ", iter, " loss: ", self.imit_loss)
                wandb.log({"imitation_train_loss": self.imit_loss})

        return loss.detach().cpu().numpy()

    def save(self, save_path, episode):

        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.policy.state_dict(), model_actor_path)

    def load(self, file):
        self.policy.load_state_dict(torch.load(file))






