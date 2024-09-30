import copy
import numpy as np
from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int

class HSSGGTrajectoryPool:
	def __init__(self, args, pool_length):
		self.args = args
		self.length = pool_length

		self.pool = []
		self.pool_obs = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state, obs_trajectory):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_obs.append(obs_trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_obs[self.counter%self.length] = obs_trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_obs), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_obs = copy.deepcopy(self.pool_obs)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_obs += copy.deepcopy(self.pool_obs)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]),copy.deepcopy(pool_obs[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class HSSGGMatchSampler:
	def __init__(self, args, achieved_trajectory_pool):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
		self.delta = self.env.distance_threshold
		self.goal_distance = get_goal_distance(args)

		self.length = args.episodes
		init_goal = self.env.reset()['achieved_goal'].copy()
		self.pool = [] #np.tile(init_goal[np.newaxis,:],[self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		self.init_state = self.env.reset()['observation'].copy()

		self.match_lib = gcc_load_lib('learner/cost_flow.c')
		self.achieved_trajectory_pool = achieved_trajectory_pool

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.reset()
			dis = self.goal_distance(obs['achieved_goal'],obs['desired_goal'])
			if dis>self.max_dis: self.max_dis = dis

	def add_noise(self, pre_ss_goal, noise_std=None):
		ss =  pre_ss_goal[0].copy()
		goal = pre_ss_goal[1].copy()
		if noise_std is None: noise_std = self.delta
		noise_goal = np.random.normal(0, noise_std, size=len(goal))
		noise_goal[-1] = 2*np.abs(noise_goal[-1])
		if 'Slide-v3' in self.args.env :
			noise_goal[-1] = 0
		goal += noise_goal
		ss += np.random.normal(0, noise_std, size=len(ss))
		return ss.copy(),goal.copy()

	def sample(self, idx):
		return self.add_noise(self.pool[idx])


	def find(self, goal):
		res = np.sqrt(np.sum(np.square(self.pool-goal),axis=1))
		idx = np.argmin(res)
		# if test_pool:
		# 	self.args.logger.add_record('Distance/sampler', res[idx])
		return self.pool[idx].copy()

	def update(self, initial_goals, initial_obs, desired_goals):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy([(obs,goal) for obs in initial_obs for goal in desired_goals])
			return

		achieved_pool, obs_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		candidate_ssg = []
		candidate_edges = []
		candidate_id = []

		agent = self.args.agent
		achieved_value = []
		for i in range(len(achieved_pool)):
			obs = [ goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for  j in range(achieved_pool[i].shape[0])]
			feed_dict = {
				agent.raw_obs_ph: obs
			}
			value = agent.sess.run(agent.q_pi, feed_dict)[:,0]
			value = np.clip(value, -1.0/(1.0-self.args.gamma), 0)
			achieved_value.append(value.copy())

		n = 0
		graph_id = {'achieved':[],'desired':[]}
		for i in range(len(achieved_pool)):
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)):
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)):
			for j in range(len(desired_goals)):
				goalres = np.sqrt(np.sum(np.square(achieved_pool[i]-desired_goals[j]),axis=1)) - achieved_value[i]/(self.args.hgg_L/self.max_dis/(1-self.args.gamma))
				traj = np.sqrt(np.sum(np.square(achieved_pool[i]),axis=1))
				diff = np.diff(traj)
				tolerance = 1e-5
				critical_index = np.where(np.abs(diff) > tolerance)[0][0] + 1
				obres = ((1-self.args.hssgg_beta)*np.sqrt(np.sum(np.square(obs_pool[i]-initial_obs[j]),axis=1)) 
					  + (self.args.hssgg_beta)*np.sqrt(np.sum(np.square(obs_pool[i]-obs_pool[i][critical_index]),axis=1)))
				match_dis = np.min(goalres)+np.min(obres)*self.args.hgg_c
				match_g_idx = np.argmin(goalres)
				match_ss_idx = np.argmin(obres)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_ssg.append((obs_pool[i][match_ss_idx],achieved_pool[i][match_g_idx]))
				candidate_edges.append(edge)
				candidate_id.append(j)
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)
		assert match_count==self.length

		explore_ssg = [0]*self.length
		for i in range(len(candidate_ssg)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_ssg[candidate_id[i]] = copy.deepcopy(candidate_ssg[i])
		assert len(explore_ssg)==self.length
		self.pool = copy.deepcopy(explore_ssg)

class HSSGGLearner:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.goal_distance = get_goal_distance(args)

		self.env_List = []
		for i in range(args.episodes):
			print('############################ env: ',i,'#########################')
			if i==0:
				self.env_List.append(make_env(args,render_mode='human'))
			else:
				self.env_List.append(make_env(args))

		self.achieved_trajectory_pool = HSSGGTrajectoryPool(args, args.hgg_pool_size)
		self.sampler = HSSGGMatchSampler(args, self.achieved_trajectory_pool)
	def learn(self, args, env, env_test, agent, buffer):
		# import time
		initial_goals = []
		initial_obs = []
		desired_goals = []
		for i in range(args.episodes):
			obs = self.env_List[i].reset()
			goal_a = obs['achieved_goal'].copy()
			goal_d = obs['desired_goal'].copy() 
			obs = obs['observation'].copy() 
			initial_goals.append(goal_a.copy())
			initial_obs.append(obs.copy())
			desired_goals.append(goal_d.copy())

		self.sampler.update(initial_goals, initial_obs, desired_goals)

		achieved_trajectories = []
		achieved_obs_trajectories = []
		achieved_init_states = []
		for i in range(args.episodes):
			obs = self.env_List[i].get_obs()
			init_state = obs['observation'].copy()
			explore_initial_state, explore_goal = self.sampler.sample(i)
			self.env_List[i].initial_state = explore_initial_state.copy()
			self.env_List[i].goal = explore_goal.copy()
			obs = self.env_List[i].get_obs()
			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]
			obs_trajectory = [obs['observation'].copy()]
			for timestep in range(args.timesteps):
				action = agent.step(obs, explore=True)
				obs, reward, done, info = self.env_List[i].step(action)
				trajectory.append(obs['achieved_goal'].copy())
				obs_trajectory.append(obs['observation'].copy())
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break
			achieved_trajectories.append(np.array(trajectory))
			achieved_obs_trajectories.append(np.array(obs_trajectory))
			achieved_init_states.append(init_state)
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				args.stc_act = 0.075
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				agent.target_update()

		selection_trajectory_idx = {}
		for i in range(self.args.episodes):
			if self.goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:
				selection_trajectory_idx[i] = True
		for idx in selection_trajectory_idx.keys():
			self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy(), achieved_obs_trajectories[idx])
		print('number of valid trajectories collected :',len(self.achieved_trajectory_pool.pool))
