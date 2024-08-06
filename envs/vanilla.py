import gymnasium as gym
import numpy as np
from envs.utils import goal_distance, goal_distance_obs
from utils.os_utils import remove_color

class VanillaGoalEnv():
	def __init__(self, args,render_mode='human'):
		self.args = args
		self.env = gym.make(args.env,render_mode=render_mode)
		self.np_random = self.env.np_random

		self.distance_threshold = self.env.distance_threshold

		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.max_episode_steps = self.env._max_episode_steps

		self.fixed_obj = False
		self.has_object = self.env.has_object
		self.obj_range = self.env.obj_range
		self.target_range = self.env.target_range
		self.target_offset = self.env.target_offset
		self.target_in_the_air = self.env.target_in_the_air
		if self.has_object: self.height_offset = self.env.height_offset

		self.render = self.env.render
		self.get_obs = self.env.get_obs
		self.reset_sim = self.env.reset

		self.reset_ep()
		self.env_info = {
			'Rewards': self.process_info_rewards, # episode cumulative rewards
			'Distance': self.process_info_distance, # distance in the last step
			'Success@green': self.process_info_success # is_success in the last step
		}

	def compute_reward(self, achieved, goal):
		dis = goal_distance(achieved[0], goal)
		return -1.0 if dis>self.distance_threshold else 0.0

	def compute_distance(self, achieved, goal):
		return np.sqrt(np.sum(np.square(achieved-goal)))

	def process_info_rewards(self, obs, reward, info):
		self.rewards += reward
		return self.rewards

	def process_info_distance(self, obs, reward, info):
		return self.compute_distance(obs['achieved_goal'], obs['desired_goal'])

	def process_info_success(self, obs, reward, info):
		return info['is_success']

	def process_info(self, obs, reward, info):
		return {
			remove_color(key): value_func(obs, reward, info)
			for key, value_func in self.env_info.items()
		}

	def step(self, action):
		# imaginary infinity horizon (without done signal)
		obs, reward, terminated, truncated, info = self.env.step(action)
		done = terminated or truncated
		info = self.process_info(obs, reward, info)
		reward = self.compute_reward((obs['achieved_goal'],self.last_obs['achieved_goal']), obs['desired_goal'])
		self.last_obs = obs.copy()
		return obs, reward, False, info

	def reset_ep(self):
		self.rewards = 0.0

	def reset(self):
		self.reset_ep()
		obs, _ = self.env.reset()
		self.last_obs = obs.copy()
		return self.last_obs.copy()

	@property
	def sim(self):
		return self.env.sim
	@sim.setter
	def sim(self, new_sim):
		self.env.sim = new_sim

	# @property
	# def initial_state(self):
	# 	return self.env.initial_state

	# @initial_state.setter()
	# def initial_state(self, value):
	# 	self.env.initial_state = value.copy()

	@property
	def initial_gripper_xpos(self):
		return self.env.initial_gripper_xpos.copy()

	@property
	def goal(self):
		return self.env.goal.copy()
	@goal.setter
	def goal(self, value):
		self.env.goal = value.copy()
