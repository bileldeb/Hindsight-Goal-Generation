
import numpy as np
import time
from common import get_args,experiment_setup
import panda_gym
import gymnasium
import imageio
from envs import make_env


args = get_args()
env, env_test, agent, buffer, learner, tester = experiment_setup(args)
agent.load_network('checkpoints/pap_task/modified_hgg/pretrained.bilel')
env = make_env(args,render_mode='rgb_array')


def render(name = 'MobileSlide - hssgg'):
    images = []
    observation = env.reset()
    images.append(env.render())
    print('rendering')
    for _ in range(250):
        action = agent.step(observation, explore=False)
        observation, reward, terminated, info = env.step(action)
        images.append(env.render())
        if terminated :
            observation, info = env.reset()
            images.append(env.render())
            break
    imageio.mimsave(name+'.gif', images)


render()