import sys
sys.path.insert(0, '/home/ak/Projects/spinningup')

from spinup import ppo
import tensorflow as tf
import gym

env_fn = lambda : gym.make('CartPole-v1')

logger_kwargs = dict(output_dir='/home/Projects/runSpinup/output_dir', exp_name='cp_ppo')

ppo(env_fn=env_fn, logger_kwargs=logger_kwargs)

