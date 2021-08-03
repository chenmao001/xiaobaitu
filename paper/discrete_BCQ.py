import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from collections import namedtuple
from itertools import count
import debug as db
import pathlib
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

Transition =namedtuple('Transition',  ('state', 'action', 'next_state', 'reward', 'done'))


# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions):
		super(Conv_Q, self).__init__()
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

		self.q1 = nn.Linear(3136, 512)
		self.q2 = nn.Linear(512, num_actions)

		self.i1 = nn.Linear(3136, 512)
		self.i2 = nn.Linear(512, num_actions)


	def forward(self, state):
		c = F.relu(self.c1(state))
		c = F.relu(self.c2(c))
		c = F.relu(self.c3(c))

		q = F.relu(self.q1(c.reshape(-1, 3136)))
		i = F.relu(self.i1(c.reshape(-1, 3136)))
		i = self.i2(i)
		return self.q2(q), F.log_softmax(i, dim=1), i


# Used for Box2D / Toy problems  ,q网络和监督生成网络，输入状态，返回q值，策略pi（各个动作概率值），概率在softmax前的值
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, 256)
		self.q2 = nn.Linear(256, 256)
		self.q3 = nn.Linear(256, num_actions)

		self.i1 = nn.Linear(state_dim, 256)
		self.i2 = nn.Linear(256, 256)
		self.i3 = nn.Linear(256, num_actions)		


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = F.relu(self.i3(i))  # 监督学习，输出策略pi的概率值
		return self.q3(q), F.log_softmax(i, dim=1), i


#  特征期望近似网络,输入状态和动作,输出特征期望
class Feature_Expectation(nn.Module):
	def __init__(self, state_dim, action_dim):  # 初始化输入状态维度,动作维度,特征维度为二者和
		super(Feature_Expectation, self).__init__()
		self.q1 = nn.Linear(state_dim+action_dim, 256)
		self.q2 = nn.Linear(256, 256)
		self.q3 = nn.Linear(256, state_dim)	  # +action_dim只考虑状态维度


	def forward(self, state, action):
		q = F.relu(self.q1(torch.cat((state,action.double()), 1)))  # 拼接状态和动作作为输入
		q = F.relu(self.q2(q))

		return F.tanh(self.q3(q))

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight.data,0.0,0.1)
		nn.init.constant_(m.bias.data,0)

class discrete_BCQ(object):
	def __init__(
		self,
			env,
			name,
		is_atari,
		num_actions,
		state_dim,
		device,
		plot,
		BCQ_threshold=0.1,  # 大于这个 概率的动作才有可能被选择，然后再判断Q 值
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=100,  # 8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
		delta = 0.01,   # 特征期望网络的终止条件
	):
	
		save_path = 'vids/%s/' % name
		pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
		self.plot = plot
		self.device = device
		self.env = env
		
		self.best_rwd = -float('inf')
		self.BATCH_SIZE = 128
		self.num_episodes = 1000  # 1000
		self.name = name
		self._delta = delta
		self._max_timesteps = 10**3
		# Determine network type
		self.Q = Conv_Q(state_dim, num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q.apply(weights_init)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.best_model = copy.deepcopy(self.Q)

		self.F = Feature_Expectation(state_dim, 1).to(self.device)   # 目前都是离散动作,动作维度为1
		self.F.apply(weights_init)
		self.F_target = copy.deepcopy(self.F)
		self.F_optimizer = getattr(torch.optim, optimizer)(self.F.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Threshold for "unlikely" actions
		self.threshold = BCQ_threshold

		# Number of training iterations
		self.iterations = 0


	def select_action(self, state, eval=False, best=False):
		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if eval:
			random_value = 1.0
		else:
			random_value = np.random.uniform(0,1)
		if  random_value> self.eval_eps:
			with torch.no_grad():
				state = torch.DoubleTensor(state).reshape(self.state_shape).to(self.device)
				if best:
					q, imt, i = self.best_model(state)
				else:
					q, imt, i = self.Q(state)
				imt = imt.exp()
				
				imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
				# Use large negative number to mask actions from argmax
				# print(type(imt),type(q),'--------------------------',imt)
				# print(imt.shape,q.shape)
				# print((imt * q + (1. - imt) * -1e8).argmax(1))  #[128,]
				# print((imt * q + (1. - imt) * -1e8).argmax(1),"--------------------------------")
				return (imt * q + (1. - imt) * -1e8).argmax(1)
		else:
			return np.random.randint(self.num_actions)

	
	def transfer_buffer(self, transitions):
		batch = Transition(*zip(*transitions))
		state = torch.cat(batch.state)
		# print(state.size(),state)  # [128, 4]
		action = torch.cat(batch.action)
		next_state = torch.cat(batch.next_state)
		# print(batch.reward)
		reward = torch.tensor(batch.reward)
		done = torch.cat(batch.done)
		# print("rewad.shape:",reward.shape,done.shape,'-----------------------------------')
		return state, action, next_state, reward, done

	def train(self, replay_buffer, rwd_weight):
		for i_episode in tqdm.tqdm(range(self.num_episodes)):
			# Sample replay buffer
			# state, action, next_state, reward, done = replay_buffer.sample()
			transitions =replay_buffer.sample(self.BATCH_SIZE)
			state, action, next_state, reward, done = self.transfer_buffer(transitions)

			if rwd_weight is None:
				x, x_dot, theta, theta_dot = next_state
				r1 = (self.env.unwrapped.x_threshold - abs(x)) / self.env.unwrapped.x_threshold - 0.8
				r2 = (self.env.unwrapped.theta_threshold_radians - abs(
					theta)) / self.env.unwrapped.theta_threshold_radians - 0.5
				reward = torch.tensor([r1 + r2])
			else:
				# feat = torch.cat((state, action.double().view(-1, 1)), 1)
				feat=state
				# reward = rwd_weight.t() @ feat
				# rwd_weight = torch.randn(4,1)   # 使用随机值并不会收敛
				reward = feat @ rwd_weight

			# print(reward,'--------------------------------------------------')
			# Compute the target Q value
			with torch.no_grad():
				q, imt, i = self.Q(next_state)
				imt = imt.exp()
				imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()  # 判断生成概率是否大于阈值，结果1或0

				# Use large negative number to mask actions from argmax
				next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)  # 大于阈值则选取q值最大的动作，小于随机选？

				q, imt, i = self.Q_target(next_state)
				target_Q = reward + (1-done) * self.discount * q.gather(1, next_action).reshape(-1, 1)

			# Get current Q estimate
			current_Q, imt, i = self.Q(state)
			current_Q = current_Q.gather(1, action)

			# Compute Q loss
			q_loss = F.smooth_l1_loss(current_Q, target_Q)
			i_loss = F.nll_loss(imt, action.reshape(-1))  # 当计算了softmax后，nll（负的最大似然neg log likelyhood)与交叉熵等同，交叉熵自动做softmax

			Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()  # 最后一项是regular

			# Optimize the Q
			self.Q_optimizer.zero_grad()
			Q_loss.backward()
			self.Q_optimizer.step()

			# Update target network by polyak or full copy every X iterations.
			self.iterations += 1
			self.maybe_update_target()

			# Do not test the model until we have been through at least 100,因为训练完一次策略会需要返回奖励和特征

			# if i_episode > self.num_episodes-2:
			# 	policy_rwd = self.testModel(self.Q)
			# 	db.printInfo('Policy Reward: %d' % policy_rwd)

		# Done training.
		print('Complete')
		pathlib.Path('plts/').mkdir(parents=True, exist_ok=True)
		plt.savefig('plts/train-%s.png' % self.name)
		if self.plot:
			self.env.render()
			self.env.close()
			plt.ioff()
			plt.show()

	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			self.Q_target.load_state_dict(self.Q.state_dict())

	def featurefn_1(self, state):
		#
		# Have to normalize everything
		# normalizer = torch.tensor([self.env.x_threshold, self.env.x_threshold, self.env.theta_threshold_radians, self.env.theta_threshold_radians])
		
		x, x_dot, theta, theta_dot = state
		x = (x + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
		#
		# Assume that the velocity never goes too high.
		x_dot = (x_dot + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
		theta = (theta + self.env.unwrapped.theta_threshold_radians) / (2 * self.env.unwrapped.theta_threshold_radians)
		theta_dot = (theta_dot + self.env.unwrapped.theta_threshold_radians) / (
					2 * self.env.unwrapped.theta_threshold_radians)
		feat = torch.tensor(
			[
				x, x_dot, theta, theta_dot,
				# x ** 2, x_dot ** 2, theta ** 2, theta_dot ** 2,
			]
		)
		return feat


	def featurefn_2(self, state):
		#
		# Have to normalize everything
		# normalizer = torch.tensor([self.env.x_threshold, self.env.x_threshold, self.env.theta_threshold_radians, self.env.theta_threshold_radians])
		x, x_dot, theta, theta_dot = state[:, 0],state[:, 1], state[:, 2], state[:, 3],

		x = (x + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
		# print(x.size())
		# Assume that the velocity never goes too high.
		x_dot = (x_dot + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
		theta = (theta + self.env.unwrapped.theta_threshold_radians) / (2 * self.env.unwrapped.theta_threshold_radians)
		theta_dot = (theta_dot + self.env.unwrapped.theta_threshold_radians) / (
					2 * self.env.unwrapped.theta_threshold_radians)
		feat = torch.cat(
			(x.view(1, -1), x_dot.view(1, -1), theta.view(1, -1), theta_dot.view(1, -1)),0
			# (x ** 2).view(1, -1), (x_dot ** 2).view(1, -1), (theta ** 2).view(1, -1), (theta_dot ** 2).view(1, -1)), 0
		)
		return feat

	# 返回测试的平均奖励和特征
	def gatherAverageFeature(self, best=False):
		with torch.no_grad():
			n_iter = 100
			rwd_sum = None
			for i in tqdm.tqdm(range(n_iter)):				
				rwd = self.testModel(self.Q, ifbest=best)
				if rwd_sum is None:
					rwd_sum = rwd
				else:
					rwd_sum += rwd
			rwd_sum /= n_iter
			db.printInfo(rwd_sum)
		return rwd_sum


	def testModel(self, mdl, save_states=False, ifbest=False):
		ep_rwd = 0
		state_list = []
		state_tp = self.env.reset()
		state = torch.from_numpy(state_tp).unsqueeze(0).to(self.device, dtype=torch.double)
		if save_states:
			state_list.append(self.featurefn_1(state_tp))
		with torch.no_grad():
			for t in count():
				# a = self.select_action(state_tp)  # 根据当前的Q网络选择动作
				with torch.no_grad():
					state = torch.DoubleTensor(state_tp).reshape(self.state_shape).to(self.device)
					# q, imt, i = self.Q(state)
					# a = q.max(1)[1].view(1, 1)
					a = self.select_action(state, best=ifbest)
				
				state_tp, reward, done, _ = self.env.step(int(a))
				state = torch.from_numpy(state_tp).unsqueeze(0).to(self.device, dtype=torch.float)
				if save_states:
					state_list.append(self.featurefn_1(state_tp))
				# ep_rwd += reward
				ep_rwd = t
				if done or t > 30000:
					self.showProgress(ep_rwd)
					break
		#
		# Based on the total reward for the episode determine the best model.
		if ep_rwd > self.best_rwd:  # and not save_states:
			self.best_rwd = ep_rwd
			self.best_model = copy.deepcopy(mdl)
		if not save_states:
			return ep_rwd
		else:
			return ep_rwd, state_list
			
	def showProgress(self, ep_rwd):
		plt.figure(2)
		plt.clf()
		plt.title('Performance: %s' % self.name)
		plt.xlabel('Episode')
		plt.ylabel('Duration')
		plt.plot(ep_rwd)
	
	def train_feaexp(self, replay_buffer, s_init):

		for t in count():
			# Sample replay buffer
			transitions =replay_buffer.sample(self.BATCH_SIZE, train=True, val=False)   # 分割训练集验证集
			state, action, next_state, reward, done = self.transfer_buffer(transitions) 
			next_action = self.select_action(next_state, eval=True).view(-1, 1)  # 多个状态动作对
			
			# Compute the target mu value
			with torch.no_grad():
				# print(next_state.shape,next_action.shape,'------------------------------------------------')
				
				mu = self.F(next_state, next_action)
				# print(((1-done.view(-1,1)) * self.discount * mu).shape)
				target_mu = state + (1-done.view(-1,1)) * self.discount * mu  # torch.cat((next_state, next_action.double()), 1)只考虑状态

			# Get current mu estimate
			current_mu = self.F(state, action)

			# Compute mu loss
			mu_loss = F.mse_loss(current_mu, target_mu)  # l1 loss,后面再尝试mse loss

			# Optimize the F
			self.F_optimizer.zero_grad()
			mu_loss.backward()
			self.F_optimizer.step()

			# Update target network by polyak or full copy every X iterations.
			if t % self.target_update_frequency == 0:
				self.F_target.load_state_dict(self.F.state_dict())

			if t % 100 == 0:
                # logger.log("been trained {} steps".format(t))
				transitions =replay_buffer.sample(self.BATCH_SIZE, train=False, val=True)   # 分割训练集验证集
				state, action, next_state, reward, done = self.transfer_buffer(transitions)	
				next_action = self.select_action(next_state, eval=True).view(-1,1)
                # 论文中的公式（2）
				mu_est_val = self.F(state, action)
				mu_target_val = state/state.norm() + self.discount * self.F(next_state, next_action)  #torch.cat((state, action.double()), 1)只考虑状态
                # average over rows and cols
				td_errors_val = ((mu_est_val - mu_target_val)**2).mean(0)  # 均方误差
				if td_errors_val.norm().item() < self._delta:  # 二范数
					# logger.log("mean validation td_errors: {}".format(td_errors_val))
					print('t----------------------------------',t)
					break
			if t > self._max_timesteps:
				print('t----------------------------------',t)
				break
		mus = None	
		for state in s_init:
			action = self.select_action(state, eval=True)
			mu = self.F(state, action.view(-1, 1))
			if mus is None:
				mus= mu
			else:
				mus+=mu
		mus = mus/len(s_init)
		return 	 mus/((mus).norm(2))
		# demo_num = 100
		# feature_expectations = None
		# for _ in range(demo_num):
		# 	state_tp = self.env.reset()
		# 	# print(type(state_tp),"---------------------------------")
		# 	state = torch.from_numpy(state_tp).unsqueeze(0).to(self.device, dtype=torch.double)
		# 	# print(type(state),"---------------------------------")
		# 	demo_length = 0
		# 	done = False
			
		# 	while not done:
		# 		demo_length += 1
		# 		a = self.select_action(state, best=False)
		# 		state_tp, reward, done, _ = self.env.step(int(a))
		# 		state = torch.DoubleTensor(state_tp).reshape(self.state_shape).to(self.device)
				
		# 		features = state.squeeze()
		# 		if feature_expectations is None:
		# 			feature_expectations = features
		# 		else:
		# 			feature_expectations += (self.discount**(demo_length)) * np.array(features)

		# feature_expectations = feature_expectations/ demo_num
		# # print(feature_expectations,type(feature_expectations))
		# return feature_expectations/feature_expectations.norm(2)
