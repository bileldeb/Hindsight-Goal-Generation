import numpy as np
import time
from common import get_args,experiment_setup
import panda_gym
import gymnasium


if __name__=='__main__':
	args = get_args()
	env, env_test, agent, buffer, learner, tester = experiment_setup(args)
	agent.load_network('checkpoints/mobile_reach_task_hgg/checkpoint.bilel')

	args.logger.summary_init(agent.graph, agent.sess)

	# Progress info
	args.logger.add_item('Epoch')
	args.logger.add_item('Cycle')
	args.logger.add_item('Episodes@green')
	args.logger.add_item('Timesteps')
	args.logger.add_item('TimeCost(sec)')

	# Algorithm info
	for key in agent.train_info.keys():
		args.logger.add_item(key, 'scalar')

	# Test info
	for key in tester.info:
		args.logger.add_item(key, 'scalar')

	args.logger.summary_setup()

	# max_episode_length = 1000
	# min_episode_length = 500 

	args.timesteps = 350

	for epoch in range(args.epochs):
		p = epoch / args.epochs
		args.hssgg_beta = 0.75-3*p/4
		# episode_length = (1-p)*min_episode_length + p*max_episode_length
		# episode_length = int(episode_length)
		# print('Epoch:',epoch,'| Number of timesteps per episode:',episode_length)
		# args.timesteps = episode_length
		for cycle in range(args.cycles):
			args.logger.tabular_clear()
			args.logger.summary_clear()
			start_time = time.time()

			learner.learn(args, env, env_test, agent, buffer)
			tester.cycle_summary()

			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epochs))
			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
			args.logger.add_record('Episodes', buffer.counter)
			args.logger.add_record('Timesteps', buffer.steps_counter)
			args.logger.add_record('TimeCost(sec)', time.time()-start_time)

			args.logger.tabular_show(args.tag)
			args.logger.summary_show(buffer.counter)

		tester.epoch_summary()

	tester.final_summary()
	agent.save_network('checkpoints/mobile_slide_task/pretrained_hgg/checkpoint.bilel')
