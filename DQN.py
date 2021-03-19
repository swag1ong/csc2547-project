# Copyright (c) 2021, Programmable digital systems group, University of Toronto
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
# Change stable_baseline to stable_basleine3 if you are using the newer version
from ABC_Env_CSC2547 import ABCEnv
import gym 
import numpy as np

env = ABCEnv()

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Instantiate the env
env = ABCEnv()
# wrap it
env = make_vec_env(lambda: env, n_envs=1)

# Train the agent
"""
Something you might want to play around with, learning_rate, total timesteps etc.. 
Always choose a sample efficient algorithm
"""
total_timesteps = 200
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./CSC2547_tensorboard/")
model.learn(total_timesteps)

model_name = "DQN_timesteps_" + str(total_timesteps)
model.save(model_name)

model.load(model_name, env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes = 2)
print("mean_reward is: ", mean_reward)
print("std_reward is: ", std_reward)
