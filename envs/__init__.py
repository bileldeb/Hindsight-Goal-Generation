import gymnasium as gym
from .utils import goal_distance, goal_distance_obs
from .vanilla import VanillaGoalEnv

Robotics_envs_id = [
	'PandaMobilePickAndPlace-v3',
	'PandaMobileReach-v3',
	'PandaReach-v3',
]

def make_env(args,render_mode='rgb_array'):
	return VanillaGoalEnv(args,render_mode=render_mode)

def clip_return_range(args):
	gamma_sum = 1.0/(1.0-args.gamma)
	return {
		'PandaMobilePickAndPlace-v3':(-gamma_sum, 0.0),
		'PandaMobileReach-v3':(-gamma_sum, 0.0),
		'PandaReach-v3':(-gamma_sum, 0.0),
	}[args.env]

