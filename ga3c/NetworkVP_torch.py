# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:51:40 2017

@author: valeodevbox
"""

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

import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T

from Config import Config

from collections import namedtuple
SavedAction = namedtuple('SavedAction', ['idx', 'action', 'value'])

class Policy(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.affine1 = nn.Linear(2592, 256)
        self.pi = nn.Linear(256, num_actions)
        self.v = nn.Linear(256, 1)

        self.saved_rnn_states = []
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 2592)
        x = F.relu(self.affine1(x))
        p = self.pi(x)
        v = self.v(x)
        return F.softmax(p), v


class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON      

        self.model = Policy(in_channels=4, num_actions=num_actions)
        self.model.cuda()
        self.opt = optim.Adam(self.model.parameters(), lr=1e-4)


    def _create_graph(self):
        return Exception('not implemented')

    def reset_state(self, idx):
        return Exception('not implemented')
        
    def predict_action(self, x, idx):
        x = np.moveaxis(x, 3, 1)
        state = torch.from_numpy(x).float().cuda()
        probs, state_value = self.model(Variable(state))
        action = probs.multinomial()
        self.model.saved_actions.append(SavedAction(idx, action, state_value)) 
        return action.cpu().data.numpy()

    def predict_p_and_v(self, x):
        x = np.moveaxis(x, 3, 1)
        state = torch.from_numpy(x).float().cuda()
        p, v = self.model(Variable(state))
        return p.cpu().data.numpy(), v.cpu().data.numpy()
    
    def train(self, x, y_r, a, trainer_id):     
        rewards = torch.Tensor(y_r)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        
        return Exception('not implemented')
        

    def log(self, x, y_r, a):
        return Exception('not implemented')

    def _checkpoint_filename(self, episode):
        return Exception('not implemented')
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        return Exception('not implemented')

    def load(self):
        return Exception('not implemented')
       
    def get_variables_names(self):
        return Exception('not implemented')

    def get_variable_value(self, name):
        return Exception('not implemented')


if __name__ == '__main__': 
    model = NetworkVP('0','torchmodel',4)
     
    state = np.zeros((10,84,84,4),dtype=np.float32)
    #p, v = model.predict_p_and_v(state)
    a = model.predict_action(state,np.arange(10))
      
    N = 1000
    st = time.time()
    for i in range(N):
        state = np.zeros((10,84,84,4),dtype=np.float32)
        a = model.predict_action(state,np.arange(10))
    
    print( (time.time() -st)*1000.0/N, 'ms/forward')
    