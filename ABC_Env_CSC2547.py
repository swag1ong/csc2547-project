# Copyright (c) 2021, Programmable digital systems group, University of Toronto
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 

#The reward function has been adjusted to exponential as suggested by Yufei Kang
# Reward = (Base^(1- current_lut / Initial_lut) - Omega)
# Omega = 1 for now

import numpy as np
import gym
from gym import spaces

#Inferent from DRiLLS import
import os
import re
import datetime
import numpy as np
from subprocess import check_output
from features import extract_features

#import the yaml file directly
import yaml

class ABCEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is the env created to run the logic synthesis tool abc environment
    """

    metadata = {'render.modes': ['console']}
        
    def __init__(self):
        super(ABCEnv, self).__init__()

        #Define the action space
        self.Action_space = ['rewrite','rewrite -z','refactor','refactor -z','resub','resub -z','balance']
        self.action_space_length = len(self.Action_space)
        self.observation_space_size = 9     # number of features

        #import the yaml file
        data_file = 'params.yml'

        with open(data_file, 'r') as f:
             options = yaml.load(f, Loader=yaml.FullLoader)
        self.params = options

        self.iteration = 0
        self.episode = 0
        self.sequence = ['strash']
        self.lut_6, self.levels = float('inf'), float('inf')

        self.best_known_lut_6 = (float('inf'), float('inf'), -1, -1)
        self.best_known_sequence_lut_6 = ['strash']
        self.best_known_levels = (float('inf'), float('inf'), -1, -1)
        self.best_known_sequence_levels = ['strash']

        self.log = None

        #Due to the adjusted reward function, we also need to store the initial value of lut_count and level
        self.initial_lut_6, self.initial_levels = self._run()
        self.base = 2048

        #Run one abc iteration to get the initial lut_6 and levels
        abc_command = 'read ' + self.params['design_file'] + ';'
        abc_command += ';'.join(self.sequence) + '; '
        abc_command += 'if -K ' + str(self.params['fpga_mapping']['lut_inputs']) + '; '
        abc_command += 'print_stats;'
    
        try:
            proc = check_output([self.params['abc_binary'], '-c', abc_command])
            # get reward
            self.initial_lut_6, self.initial_levels = self._get_metrics(proc)
        except Exception as e:
            print(e)


        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have 7 optimization commands in Action_space
        n_actions = len(self.params['optimizations'])
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(0.0,1000000,shape=(self.observation_space_size,),dtype = np.int32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        self.iteration = 0
        self.episode += 1
        
        #For easier testing
        self.update()

        self.lut_6, self.levels = float('inf'), float('inf')
        self.sequence = ['strash']

        # logging
        csv_name = 'log'+ str(self.episode) + '.csv' 
        log_file = os.path.join('log', csv_name)
        if not os.path.exists('log'):
            os.makedirs('log')
        if self.log:
            self.log.close()
        self.log = open(log_file, 'w')
        self.log.write('iteration, optimization, LUT-6, Levels, best LUT-6, best levels\n')

        state, _ = self._run()

        # logging
        self.log.write(', '.join([str(self.iteration),self.sequence[-1], str(int(self.lut_6)), str(int(self.levels))]) + '\n')
        self.log.flush()
        
        return np.array(state).astype(np.int32)

    def _run(self):
        """
        run ABC on the given design file with the sequence of commands
        """
        self.iteration += 1
        output_design_file = 'getFeature.v'
   
        abc_command = 'read ' + self.params['design_file'] + ';'
        abc_command += ';'.join(self.sequence) + '; '
        abc_command += 'write ' + output_design_file + '; '
        abc_command += 'if -K ' + str(self.params['fpga_mapping']['lut_inputs']) + '; '
        abc_command += 'print_stats;'
    
        try:
            proc = check_output([self.params['abc_binary'], '-c', abc_command])
            # get reward
            lut_6, levels = self._get_metrics(proc)
            reward = self._get_reward(lut_6, levels) 
            self.lut_6, self.levels = lut_6, levels
            # get new state of the circuit
            state = self._get_state(output_design_file)
            return state, reward
        except Exception as e:
            print(e)
            return None, None


    def _get_state(self, design_file):
        return extract_features(design_file, self.params['yosys_binary'], self.params['abc_binary'])

    def _get_metrics(self, stats):
        """
        parse LUT count and levels from the stats command of ABC
        """
        line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
        ob = re.search(r'lev *= *[0-9]+', line)
        levels = int(ob.group().split('=')[1].strip())
        
        ob = re.search(r'nd *= *[0-9]+', line)
        lut_6 = int(ob.group().split('=')[1].strip())

        return lut_6, levels

    def _get_reward(self, lut_6, levels):
        """
        Mark reward as only the area difference
        The reward function has been adjusted to exponential as suggested by Yufei Kang
        Reward = (Base^(1- current_lut / Initial_lut) - Omega)
        Omega = 0 for now
        """
        # Calculate the area difference
        reward = 1 - (lut_6 / self.initial_lut_6)
        #Make it exponential
        reward = np.power(self.base, reward) - 1 
        # now calculate the reward
        return reward
   
    def step(self, action):
        """
        accepts optimization index and returns (new state, reward, done, info)
        """
        self.sequence.append(self.params['optimizations'][action])
        new_state, reward = self._run()
        
        # logging
        if self.lut_6 <= self.best_known_lut_6[0]:
            self.best_known_lut_6 = (int(self.lut_6), int(self.levels), self.episode, self.iteration)
            self.best_known_sequence_lut_6 = self.sequence
        if self.levels <= self.best_known_levels[1]:
            self.best_known_levels = (int(self.lut_6), int(self.levels), self.episode, self.iteration)
            self.best_known_sequence_levels = self.sequence
        self.log.write(', '.join([str(self.iteration), self.sequence[-1], str(int(self.lut_6)), str(int(self.levels))]) + ', ' +
            '; '.join(list(map(str, self.best_known_lut_6))) + ', ' + 
            '; '.join(list(map(str, self.best_known_levels))) + '\n')
        self.log.flush()

        # Optionally we can pass additional info, we are not using that for now
        info = {'episodes': self.episode,'iterations':self.iteration,'Best_LUT':self.best_known_lut_6,'LUT':self.lut_6,'Level':self.levels}
        
        #Done definition
        done = bool(self.iteration == (self.params['iterations']))

        return np.array(new_state).astype(np.int32), reward, done, info



    def render(self, mode='console'):
        if mode != 'console':
           raise NotImplementedError()

    def close(self):
        pass

    def update(self):
        print("Episode ", self.episode,": ","Best Area is",self.best_known_lut_6)
        print("The corresponding opt command sequence is ",self.best_known_sequence_lut_6)