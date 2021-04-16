# -*- coding: utf-8 -*-
import os
import gym
import math
import copy
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import tqdm
import random
import pathlib
import argparse
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use("TkAgg")
import debug as db
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from itertools import count
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from collections import namedtuple
from discrete_BCQ import discrete_BCQ

import warnings

warnings.filterwarnings("ignore")
torch.set_default_tensor_type(torch.DoubleTensor)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def reset(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, train=False, val=False):
        if train:
            return random.sample(self.memory[:int(self.position*0.7)], batch_size)
        if val:
            return random.sample(self.memory[int(self.position*0.7):self.position], batch_size)
  
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(100000)
HIDDEN_LAYER = 64  # NN hidden layer size


# DQN网络，隐层固定为64，一共四层，输入输出维度专为CartPole环境，因此固定为4和2；forward传入状态，输出对应动作个数的奖励
class DQN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l1_1 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l1_2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        # self.l1_3 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l1_1(x))
        x = F.relu(self.l1_2(x))
        # x = F.relu(self.l1_3(x))
        x = self.l2(x)
        return x


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

#
class DQN_Trainer(object):
    #
    # if gpu is to be used
    device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.95
    num_episodes = 50 # 50

    EPS_END = 0.05
    EPS_DECAY = num_episodes * 0.9
    TARGET_UPDATE = 10
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    def __init__(self, args, env, name):
        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a clamped and down-scaled render buffer in get_screen()
        save_path = 'vids/%s/' % name
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        self.env = env
        # self.env = gym.wrappers.Monitor(env, save_path, video_callable=lambda episode_id: episode_id % 199 == 0)
        self.env.reset()
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.is_trained = False
        self.avgFeature = None
        if args.configStr is not None:
            self.is_trained = True
            pth = os.path.abspath(args.configStr)
            assert pathlib.Path(pth).exists()
            data = torch.load(pth)
            self.policy_net.load_state_dict(data['mdl'])
            if 'avgFeat' in data:
                self.avgFeature = data['avgFeat']
            db.printInfo('LOADED MODEL')

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.best_model = None
        self.best_rwd = -float('inf')

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        # self.memory = ReplayMemory(100000)

        self.NUM_UPDATE = 1
        self.steps_done = 0
        self.episode_durations = []
        self.plot = args.plot
        self.name = name
        plt.ion()
        if self.plot:
            plt.figure()
            self.init_screen = self.get_screen()
            plt.imshow(self.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                    interpolation='none')
            plt.title('Example extracted screen')
            # plt.show()

    def get_cart_location(self, screen_width):
        world_width = self.env.unwrapped.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.unwrapped.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(self.device)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print(self.policy_net(state).max(1))
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(memory) < self.BATCH_SIZE:
            return
        for i in range(self.NUM_UPDATE):
            transitions = memory.sample(self.BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    # 返回奖励和特征
    def testModel(self, mdl, save_states=False):
        ep_rwd = 0
        state_list = []
        state_tp = self.env.reset()
        state = torch.from_numpy(state_tp).unsqueeze(0).to(self.device, dtype=torch.double)  # dtype=torch.float)
        if save_states:
            state_list.append(self.featurefn(state_tp))
        with torch.no_grad():
            for t in count():
                a = self.policy_net(state).max(1)[1].view(1, 1)
                state_tp, reward, done, _ = self.env.step(a.item())
                state = torch.from_numpy(state_tp).unsqueeze(0).to(self.device, dtype=torch.double)  # dtype=torch.float)
                if save_states:
                    state_list.append(self.featurefn(state_tp))
                ep_rwd += reward
                if done or t > 30000:
                    break
        # print('ep_reward:', ep_rwd)
        # Based on the total reward for the episode determine the best model.
        if ep_rwd > self.best_rwd and not save_states:
            self.best_rwd = ep_rwd
            self.best_model = copy.deepcopy(mdl)
        if not save_states:
            return ep_rwd
        else:
            return ep_rwd, state_list

    # 该函数也是定制的，针对CartPole的状态，特征表示为状态和状态的平方；返回特征表示（是一个tensor)
    def featurefn(self, state):
        #
        # Have to normalize everything
        # normalizer = torch.tensor([self.env.x_threshold, self.env.x_threshold, self.env.theta_threshold_radians, self.env.theta_threshold_radians])
        x, x_dot, theta, theta_dot = state
        x = (x + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
        #
        # Assume that the velocity never goes too high.
        x_dot = (x_dot + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
        theta = (theta + self.env.unwrapped.theta_threshold_radians) / (2 * self.env.unwrapped.theta_threshold_radians)
        theta_dot = (theta_dot + self.env.unwrapped.theta_threshold_radians) / (2 * self.env.unwrapped.theta_threshold_radians)
        feat = torch.tensor(
            [
                x, x_dot, theta, theta_dot,
                # x ** 2, x_dot ** 2, theta ** 2, theta_dot ** 2,
            ]
        )
        return feat


    def train(self, rwd_weight=None):
        #
        # Train.
        for i_episode in tqdm.tqdm(range(self.num_episodes)):
            #
            # Initialize the environment and state
            state = torch.from_numpy(self.env.reset()).unsqueeze(0).to(self.device, dtype=torch.double)  # dtype=torch.float)
            for t in count():
                #
                # Select and perform an action
                action = self.select_action(state)
                next_state_np, reward, done, _ = self.env.step(action.item())
                if self.plot and i_episode % 100 == 0:
                    self.get_screen()
                next_state = torch.from_numpy(next_state_np).unsqueeze(0).to(self.device, dtype=torch.double)  # dtype=torch.float)
                if rwd_weight is None:
                    reward = torch.tensor([reward], device=self.device)
                    x, x_dot, theta, theta_dot = next_state_np
                    r1 = (self.env.unwrapped.x_threshold - abs(x)) / self.env.unwrapped.x_threshold - 0.8
                    r2 = (self.env.unwrapped.theta_threshold_radians - abs(theta)) / self.env.unwrapped.theta_threshold_radians - 0.5
                    #
                    # Must be R ∈ [-1, 1]
                    reward = torch.tensor([r1 + r2])
                else:
                    feat = self.featurefn(next_state_np)
                    reward = rwd_weight.t() @ feat
                #
                # Observe new state
                if done:
                    done = 1.
                #
                # Store the transition in self.memory
                memory.push(state, action, next_state, reward, torch.tensor([done], dtype=torch.double))
                #
                # Move to the next state
                state = next_state
                #
                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done or t > 30000:
                    self.episode_durations.append(t + 1)
                    self.showProgress(i_episode)
                    break
            #
            # Do not test the model until we have been through at least 100
            policy_rwd = 0
            if i_episode > self.num_episodes-2:  # 100
                policy_rwd = self.testModel(self.policy_net)
                db.printInfo('Policy Reward: %d' % policy_rwd)
            #
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        #
        # Done training.
        print('Complete')
        self.is_trained = True
        pathlib.Path('plts/').mkdir(parents=True, exist_ok=True)
        plt.savefig('plts/train-%s.png' % self.name)
        if self.plot:
            self.env.render()
            self.env.close()
            plt.ioff()
            plt.show()

    def showProgress(self, e_num):
        means = 0
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)  # 这个变量相当于真正的奖励
        if len(self.episode_durations) >= 100:  # 大于100之前means都输出0
            means = durations_t[-100:-1].mean().item()
        db.printInfo('Episode %d/%d Duration: %d AVG: %d'%(e_num, self.num_episodes, durations_t[-1], means))
        plt.figure(2)
        plt.clf()
        plt.title('Performance: %s' % self.name)
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if self.plot:
            # Take 100 episode averages and plot them too
            if len(durations_t) >= 100:
                means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())

            plt.pause(0.001)  # pause a bit so that plots are updated

    def saveBestModel(self):
        pathlib.Path('mdls/').mkdir(parents=True, exist_ok=True)
        state = {
            'mdl': self.best_model.state_dict(),
            'avgFeat': self.avgFeature
        }
        import datetime
        now = datetime.datetime.now()
        save_name = 'mdls/' + 'mdl_DATE-' + now.isoformat() + '.pth.tar'
        db.printInfo(save_name)
        torch.save(state, save_name)

    def phi(self, state, action):
        return state
        # return torch.cat((state, action.double()), 1)  # torch.DoubleTensor(action)

    # 针对专家数据集, 返回测试的平均奖励和特征, 专家特征也只考虑状态
    def gatherAverageFeature(self, _return_s_init=False):

        mus = []
        s_init = []
        mu = 0.0
        t = 0
        is_s_init = True
        gamma =0.99
        
        for i in range(memory.position):
            s, a, s_next, reward, done = memory.memory[t]
            if is_s_init:
                s_init.append(s)
                is_s_init = False
            mu += gamma ** t * self.phi(s, a).flatten()  # 对mu求和
            t+=1
        
            if done:
                mus.append(mu)
                mu = 0.0
                t = 0
                is_s_init = True
        
        mu_est = torch.tensor([0.0, 0.0, 0.0, 0.0 ])
        for mu in mus:
            mu_est += mu
        mu_est /= len(mus)
        mu_est/=mu_est.norm(2)
        with torch.no_grad():
            n_iter = 20  # 2000
            rwd_sum = None
            for i in tqdm.tqdm(range(n_iter)):
                rwd, states = self.testModel(self.best_model, True)   
                if rwd_sum is None:
                    rwd_sum = rwd
                else:
                    rwd_sum += rwd
            rwd_sum /= n_iter
            db.printInfo(mu_est)
            db.printInfo(rwd_sum)
        self.avgFeature = mu_est
        if _return_s_init:
            return mu_est, s_init, rwd_sum
        return mu_est, rwd_sum


    def expert_buffer(self):
        memory.reset()
        with torch.no_grad():
            while(memory.position <50000):

                # Initialize the environment and state
                state = torch.from_numpy(self.env.reset()).unsqueeze(0).to(self.device,
                                                                           dtype=torch.double)  # dtype=torch.float)
                for t in count():

                    # Select and perform an action
                    action = self.policy_net(state).max(1)[1].view(1, 1)
                    next_state_np, reward, done, _ = self.env.step(action.item())
                    next_state = torch.from_numpy(next_state_np).unsqueeze(0).to(self.device,
                                                                                 dtype=torch.double)  # dtype=torch.float)

                    # Store the transition in self.memory
                    memory.push(state, action, next_state, reward, torch.tensor([done], dtype=torch.double))
                    #
                    # Move to the next state
                    state = next_state

                    if done or t > 200:
                        break


class ALVIRL(object):

    def __init__(self, args, env):
        self.env = env

        self.expert = DQN_Trainer(args, self.env, 'Expert')
        self.device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else
        if not self.expert.is_trained:
            self.expert.train()
            mu_est, s_init, rwd_sum = self.expert.gatherAverageFeature(_return_s_init=True)
            self.s_init = s_init
            self.expert.saveBestModel()
            self.expert.expert_buffer()
        #
        # Not all saved things have this, compute just in case.
        if self.expert.avgFeature is None:
            mu_est, s_init, rwd_sum =  self.expert.gatherAverageFeature(_return_s_init=True)
            state = {
                'mdl': self.expert.policy_net.state_dict(),
                'avgFeat': self.expert.avgFeature
            }
            torch.save(state, args.configStr)
        self.expert_feat = self.expert.avgFeature
        args.configStr = None
        self.args = args

    # 算法实现完全按照Apprenticeship Learning via Inverse Reinforcement Learning
    def train(self):
        # student = DQN_Trainer(args, self.env, 'Student_0')
        student = discrete_BCQ(self.env,
                               'Student_0',
                               False,
                               self.env.action_space.n,
                               self.env.observation_space.shape[0],
                               self.device,
                               args.plot,
                               # 其余先使用默认值
                               optimizer_parameters={"lr": 3e-4},#3e-4
                               )
        # sampleFeat = student.featurefn_1(self.env.reset())  # 随机初始一个特征值[8]
        # w_0 = torch.randn(sampleFeat.size(0), 1)  # 随机初始参数w  (8,1)
        w_0 = torch.tensor([[0.5],[0.5],[0.5],[0.5]])  # 还是只针对状态吧
        # w_0 = torch.tensor([[0.1],[0.2],[0.3],[0.4]])  # 还是只针对状态吧
        w_0 /= w_0.norm(2)  # 归一化
        rwd_list = []
        t_list = []
        weights = [w_0]
        i = 1
        #
        # 测试BCQ 是否正常运行,使用真实奖励
        # for i in tqdm.tqdm(range(10)):
        #     student.train(memory, w_0)
        #     studentRwd = student.gatherAverageFeature() 
        #     bestreward = student.gatherAverageFeature(best=True)


        # Train zeroth student.
        student.train(memory, w_0)  # 训练策略pi0
        # 这个特征期望的获得居然是在线的,因此需要需要使用神经网络近似

        
            

        # to do 训练特征网络mu0
        studentFeat = student.train_feaexp(memory, self.s_init)

        studentRwd = student.gatherAverageFeature()  # 得到策略pi0的平均特征和奖励
        rwd_list.append(studentRwd)
        t_list.append((self.expert_feat - studentFeat).norm().item())  # 得到的是w1
        # 投影法简化了问题：t是w的二范数，衡量两个特征之间的距离
        # Create first student.
        weights.append((self.expert_feat - studentFeat).view(-1, 1))
        feature_bar_list = [studentFeat]  # 特征投影u-bar
        feature_list = [studentFeat]  # 特征u
        #
        # Iterate training.
        n_iter = 6  # 20
        for i in tqdm.tqdm(range(n_iter)):
            # student = DQN_Trainer(args, self.env, 'Student_%d' % (i + 1))  # 交互式地训练策略pii
            student = discrete_BCQ(self.env,
                                   'Student_%d' % (i + 1),
                                   False,
                                   self.env.action_space.n,
                                   self.env.observation_space.shape[0],
                                   self.device,
                                   args.plot,
                                   # 其余先使用默认值
                                   optimizer_parameters={"lr": 3e-4},  # 默认为-4
                                   )
            student.train(memory, weights[-1])
            studentRwd = student.gatherAverageFeature()
            studentFeat = student.train_feaexp(memory, self.s_init)

            db.printInfo("studentFeat:",studentFeat)
            
            db.printInfo("self.expert_feat:",self.expert_feat)
            

            
            rwd_list.append(studentRwd)
            feature_list.append(studentFeat)
            feat_bar_next = feature_bar_list[-1] + ((feature_list[-1] - feature_bar_list[-1]).view(-1, 1).t() @ (self.expert_feat - feature_bar_list[-1]).view(-1,1))\
                             / ((feature_list[-1] - feature_bar_list[-1]).view(-1, 1).t() @ (feature_list[-1] - feature_bar_list[-1]).view(-1,1))\
                             * (feature_list[-1] - feature_bar_list[-1])  # @矩阵乘法运算
            db.printInfo("feature_bar:",feat_bar_next)
            feature_bar_list.append(feat_bar_next)
            
            weights.append((self.expert_feat - feat_bar_next).view(-1, 1))
            t_list.append((self.expert_feat - feat_bar_next).norm().item())
            db.printInfo('t: ', t_list[-1])
        # db.printInfo(feat_bar_next)
        print('w：', weights[-1])
        # plt.figure()
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.plot(rwd_list)
        # plt.title('Average Episode Reward')
        # plt.xlabel('Student Number')
        # plt.ylabel('Episode Length')
        # plt.savefig('plts/avgRewardProgress.png')
        # plt.figure()
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.plot(t_list)
        # plt.title('L2 Policy Error')
        # plt.xlabel('Student Number')
        # plt.ylabel('Squared error of features of features')
        # plt.savefig('plts/sqerr.png')
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General tool to train a NN based on passed configuration.')
    parser.add_argument('--config', dest='configStr', default=None, type=str, help='Name of the config file to import.')
    parser.add_argument('--plot', dest='plot', default=False, action='store_true', help='Whether to plot the training progress.')

    parser.add_argument("--env", default="CartPole-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Default")  # Prepends name to filename
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--BCQ_threshold", default=0.3, type=float)  # Threshold hyper-parameter for BCQ
    parser.add_argument("--low_noise_p", default=0.2,
                        type=float)  # Probability of a low noise episode when generating buffer
    parser.add_argument("--rand_action_p", default=0.2,
                        type=float)  # Probability of taking a random action when generating buffer, during non-low noise episode
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral policy
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    seed=100
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    args = getInputArgs()
    env = gym.make('CartPole-v0')
    env.seed(0)
    arl = ALVIRL(args, env)
    arl.train()

    # memory.push(torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
    # memory.push(torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
    # memory.push(torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
    # memory.push(torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
    # memory.push(torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
    # memory.push(torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([1]), torch.tensor([1]))
    # transitions = memory.sample(3)
    # print(transitions)
    # batch = Transition(*zip(*transitions))
    # print(batch)
    # state_batch = torch.cat(batch.state)
    # print(state_batch)
    #
    # print(state_batch.view(3, 2).shape)
    # print(state_batch.view(3, 2))

    # dqnTrainer = DQN_Trainer(env, args)
    # dqnTrainer.train()
    # dqnTrainer.saveBestModel()
    # dqnTrainer.gatherAverageFeature()
