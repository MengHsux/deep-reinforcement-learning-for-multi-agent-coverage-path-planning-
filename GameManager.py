# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import gym
import pygame
from Check_Envs_GA3C_v5 import wafer_check
from Config import Config

wafer = np.loadtxt('envs.txt')
probe = np.loadtxt('probe.txt')
probe1 = np.loadtxt('probe1.txt')

location = (0, 0)


class GameManager:
    def __init__(self, display):
        # pygame.init()
        self.env = wafer_check(wafer, probe, probe1, location=location, mode=display,
                               training_time=Config.TRAINING_TIME, training_steps=Config.TRAINING_STEPS)
        # self.env = gym.make(game_name)
        self.reset()

    def reset(self):
        observation, available = self.env.reset()
        return observation, available

    def step(self, action, action1, action2):
        observation, reward, reward1, reward2, done, available, envs_mean, envs_std = self.env.step(action, action1,
                                                                                                    action2)
        return observation, reward, reward1, reward2, done, available, envs_mean, envs_std

    def get_num_actions(self):
        return self.env.action_space_num

    def get_num_state(self):
        return self.env.output.shape[0]
