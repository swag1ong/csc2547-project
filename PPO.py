# Copyright (c) 2021, Programmable digital systems group, University of Toronto
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
# Change stable_baseline to stable_basleine3 if you are using the newer version
from ABC_Env_CSC2547 import ABCEnv
import gym 
import numpy as np

env = ABCEnv()

from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env

# Instantiate the env
env = ABCEnv()
# wrap it
env = make_vec_env(lambda: env, n_envs=1)

# Train the agent
"""
Something you might want to play around with, learning_rate, total timesteps etc.. 
Always choose a sample efficient algorithm
"""
total_timesteps = 2500
model = PPO2('MlpPolicy', env, verbose=1).learn(total_timesteps)
