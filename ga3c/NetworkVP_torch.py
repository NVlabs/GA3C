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

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(1. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(1. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def init_like_torch(m):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        out_channels, in_channels = weight_shape
        stdv = 1 / np.sqrt(in_channels)
        m.weight.data.uniform_(-stdv, stdv)
        m.bias.data.uniform_(-stdv, stdv)
    elif classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        out_channels, in_channels, kh, kw = weight_shape
        print(out_channels, in_channels, kh,kw)
        stdv = 1 / np.sqrt(in_channels * kh * kw)
        m.weight.data.uniform_(-stdv, stdv)
        m.bias.data.uniform_(-stdv, stdv)
        
class Policy(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Policy, self).__init__()
        #self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        #self.affine1 = nn.Linear(2592, 256)
        
        self.affine1 = nn.Linear(16, 256)
        self.affine2 = nn.Linear(256, 256)
        
        self.pi = nn.Linear(256, num_actions)
        self.v = nn.Linear(256, 1)
        
        #self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        
        #Init
        self.apply(init_like_torch)
        #self.apply(weights_init)
        #self.pi.weight.data = normalized_columns_initializer(self.pi.weight.data, 0.01)
        #self.pi.bias.data.fill_(0)
        #self.v.weight.data = normalized_columns_initializer(self.v.weight.data, 0.01)
        #self.v.bias.data.fill_(0)
        
        #self.lstm.bias_ih.data.fill_(0)
        #self.lstm.bias_hh.data.fill_(0)

        self.saved_rnn_states = []
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x)) 
        #x = x.view(-1, 2592)
        x = x.contiguous().view(-1, 16)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        p = self.pi(x)
        v = self.v(x)
        return p, v


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
        #self.model.cuda()
        
        self.opt = optim.RMSprop(self.model.parameters(), lr=1e-4, alpha=0.99, eps=1e-1)
        #self.opt = optim.Adam(self.model.parameters(), lr=1e-4)


    def _create_graph(self):
        return Exception('not implemented')

    def reset_state(self, idx):
        return Exception('not implemented')
    '''    
    def predict_pva(self, x, idx):
        x = np.moveaxis(x, 3, 1)
        state = torch.from_numpy(x).float().cuda()
        p, v = self.model(Variable(state))
        action = p.multinomial()
        return p,v,action
    '''
    
    def predict_p_and_v(self, x, idx):
        x = np.moveaxis(x, 3, 1)
        state = torch.from_numpy(x).float()#.cuda()
        p, v = self.model(Variable(state))
        probs = F.softmax(p)
        return probs.data.numpy(), v.data.numpy()
        #return probs.cpu().data.numpy(), v.cpu().data.numpy()
    
    def train(self, x, y_r, a, idx):   
        rewards = torch.Tensor(y_r)

        rewards = Variable( rewards ) #.cuda())
        #a = Variable( torch.FloatTensor(a).cuda() )
        a = Variable( torch.ByteTensor(a.astype(np.uint8)) ) #.cuda() )
        
        #forward again (not efficient, ideally we should manage all experience history here)
        x = np.moveaxis(x, 3, 1)
        state = torch.from_numpy(x).float() #.cuda()
        p, v = self.model(Variable(state))
        
        probs = F.softmax(p)
        probs = F.relu(probs - Config.LOG_EPSILON)
        
        #log_probs = F.log_softmax(p)
        log_probs = torch.log(probs)
        

        adv = (rewards - v)
        
        log_probs_a = torch.masked_select(log_probs,a)
        #log_probs_a = torch.sum(log_probs * a, 1)            
        piloss = -torch.sum( log_probs_a * Variable(adv.data), 0)  
        entropy = torch.sum(torch.sum(log_probs*probs,1),0) * self.beta
        vloss = torch.sum(adv.pow(2),0) / 2
        
        #vloss = F.smooth_l1_loss(v, rewards ) / 2
        loss = piloss + entropy + vloss
        
        '''
        print('piloss = ',piloss.cpu().data.numpy()[0],
              'entropy = ',entropy.cpu().data.numpy()[0],
              'vloss = ',vloss.cpu().data.numpy()[0],
              'adv = ',adv.cpu().data.numpy(),
              )
        '''
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


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
    #p,v,a = model.predict_pva(state,np.arange(10))

    
      
    N = 0
    st = time.time()
    for i in range(N):
        state = np.zeros((10,84,84,4),dtype=np.float32)
        a = model.predict_action(state,np.arange(10))
    
    #print( (time.time() -st)*1000.0/N, 'ms/forward')
    