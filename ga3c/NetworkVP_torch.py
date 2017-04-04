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
from torch.nn.utils.rnn import PackedSequence
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
        
        #self.l1x = nn.Linear(256, 256)
        #self.l1h = nn.Linear(256, 256)
        
        
        self.pi = nn.Linear(256, num_actions)
        self.v = nn.Linear(256, 1)
        
        self.lstm = nn.LSTMCell(256, 256)
        
        #Init
        self.apply(init_like_torch)
        
        #self.lstm.bias_ih.data.fill_(0)
        #self.lstm.bias_hh.data.fill_(0)


    def forward(self, x, c, h):
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x)) 
        #x = x.view(-1, 2592)
        x = x.contiguous().view(-1, 16)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        #compute stateless lstm
        #h1 = x
        #c1 = h1  
        h1, c1 = self.lstm(x, (h, c))
          

        p = self.pi(h1)
        v = self.v(h1)
        return p, v, c1, h1
    

    def forward_multistep(self, x, c, h):
        x = x.contiguous().view(-1, 16)
        
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        
        
        
        z = x
        
        #Think Winnie, Think! Here, Sequence is maybe interlaced! We would need to de-interlace it first
        xt = x.view(-1, Config.TIME_MAX, 256)
        h, c = self.lstm(xt[:,0], (h, c))
        z = Variable(torch.zeros())
        for t in range(1,Config.TIME_MAX):
            h, c = self.lstm(xt[:,t], (h, c))
            z = torch.cat((z,h)) #z concat all samples, changes order of (x_agent_timestep) into (x_timestep_agent)
            
        p = self.pi(z)
        v = self.v(z)
        
        return p, v
    
    def forward2(self, x, c, h, m):
        N = m.shape[0]
        h = x.contiguous().view(-1, 16)
        h = F.relu(self.affine1(h))
        h = F.relu(self.affine2(h))

       
        hp = PackedSequence(h,m)
        
        print(hp)
        time.sleep(10000)
        #xt = x.view(-1, Config.TIME_MAX, 256)
        h, c = self.lstm(xt[:,0], (h, c))
        z = h
        for t in range(1, Config.TIME_MAX):
            h, c = self.lstm(xt[:,t], (h, c))
            z = torch.cat((z,h))
        #time.sleep(1000)
        #for i in range(0, m.size[0]):
            
        #    
        #z = Variable( torch.zeros(x.shape[0], Config.TIME_MAX))
        #h, c = self.lstm(x, (h, c))
        #for t in range(1,Config.TIME_MAX):
        #    h, c = self.lstm(x, (h, c))
            #z = torch.cat((z,h)) #z concat all samples, changes order of (x_agent_timestep) into (x_timestep_agent)
            
        
        
        
        p = self.pi(z)
        v = self.v(z)
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

    def predict_p_and_v(self, x, c, h):
        x = np.moveaxis(x, 3, 1)
        state = torch.from_numpy(x).float()#.cuda()
        c = torch.from_numpy(c).float()
        h = torch.from_numpy(h).float()
        p, v, c, h = self.model(Variable(state), Variable(c), Variable(h))
        probs = F.softmax(p)
        return probs.data.numpy(), v.data.numpy(), c.data.numpy(), h.data.numpy()
        #return probs.cpu().data.numpy(), v.cpu().data.numpy()
    
    def train(self, x, r, a, c, h, m):
        rewards = torch.Tensor(r)
        rewards = Variable( rewards ) #.cuda())
        #a = Variable( torch.ByteTensor(a.astype(np.uint8)) ) #.cuda() )
        a = Variable( torch.FloatTensor(a) )
        c = Variable(torch.from_numpy(c).float())
        h = Variable(torch.from_numpy(h).float())
        
        
        x = np.moveaxis(x, 3, 1)
        state = torch.from_numpy(x).float() #.cuda()
        p, v = self.model.forward2(Variable(state),c,h,m)
        
        probs = F.softmax(p)
        probs = F.relu(probs - Config.LOG_EPSILON)
        
        #log_probs = F.log_softmax(p)
        log_probs = torch.log(probs)
        

        print(rewards.size(), v.size())
        adv = (rewards - v)
        
        #log_probs_a = torch.masked_select(log_probs,a)
        log_probs_a = torch.sum(log_probs * a, 1)       
        
        piloss = -torch.sum( log_probs_a * Variable(adv.data), 0)  
        entropy = torch.sum(torch.sum(log_probs*probs,1),0) * self.beta
        vloss = torch.sum(adv.pow(2),0) / 2
        
        #vloss = F.smooth_l1_loss(v, rewards ) / 2
        loss = piloss + entropy + vloss
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
    #rnn version
    def train2(self, x, r, a, c, h, m):  
        #x be (N, T, C, H, W) (even padded with 0 is ok)
        #c0 be (N, 1, D)
        #h0 be (N, 1, D)
        
        #reshape x, r, a in (N*T, C, H, W)
        N = x.shape[0]
        
        #print(N, a.shape)
        
        a = a.reshape(N*Config.TIME_MAX,-1)
        r = r.reshape(N*Config.TIME_MAX,-1)
        
        mask = np.sum(a,axis=1,keepdims=True)
        #mask = Variable(torch.Tensor(mask))
        mask = Variable(torch.ByteTensor(mask.astype(np.uint8)))
        
        rewards = torch.Tensor(r)
        rewards = Variable(rewards)
        a = Variable( torch.ByteTensor(a.astype(np.uint8)) )
        #a = Variable( torch.Tensor(a) )
        
        x_ = x.reshape(-1, Config.STACKED_FRAMES, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
        x_ = np.moveaxis(x_, 3, 1)
        x_ = Variable(torch.from_numpy(x_).float())
        
        c = Variable(torch.from_numpy(c).float())
        h = Variable(torch.from_numpy(h).float())
        
        p, v = self.model.forward_multistep(x_, c, h)
        
     
        probs = F.softmax(p)
        probs = F.relu(probs - Config.LOG_EPSILON)
        
        #log_probs = F.log_softmax(p)
        log_probs = torch.log(probs) 


        adv = (rewards - v)
        adv = torch.masked_select(adv,mask)
        
        #print(adv.size())
        log_probs_a = torch.masked_select(log_probs,a) #we cannot use it because of variable length input
        #log_probs_a = torch.sum(log_probs * a, 1)  
        
        #print(log_probs_a.size(), adv.size())
        
        #time.sleep(1000)
           
        piloss = -torch.sum( log_probs_a * Variable(adv.data), 0)  
        entropy = torch.sum(torch.sum(log_probs*probs,1),0) * self.beta
        vloss = torch.sum(adv.pow(2),0) / 2
        
        #vloss = F.smooth_l1_loss(v, rewards ) / 2
        loss = piloss + entropy + vloss
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
    
        
    def train0(self, x, y_r, a, c, h, m):
        N = x.shape[0]
        #x = x.reshape(-1, Config.STACKED_FRAMES, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
        #a = a.reshape(N*Config.TIME_MAX,-1)
        #y_r = y_r.reshape(N*Config.TIME_MAX,-1)
        
        
        
        rewards = torch.Tensor(y_r)

        rewards = Variable( rewards ) #.cuda())
        #a = Variable( torch.FloatTensor(a).cuda() )
        #a = Variable( torch.ByteTensor(a.astype(np.uint8)) ) #.cuda() )
        a = Variable( torch.FloatTensor(a) )
        
        #forward again (not efficient, ideally we should manage all experience history here)
        x = np.moveaxis(x, 3, 1)
        state = torch.from_numpy(x).float() #.cuda()
        p, v, c, h = self.model(Variable(state),None,None)
        
        probs = F.softmax(p)
        probs = F.relu(probs - Config.LOG_EPSILON)
        
        #log_probs = F.log_softmax(p)
        log_probs = torch.log(probs)
        

        adv = (rewards - v)
        
        #log_probs_a = torch.masked_select(log_probs,a)
        log_probs_a = torch.sum(log_probs * a, 1)       
        
        piloss = -torch.sum( log_probs_a * Variable(adv.data), 0)  
        entropy = torch.sum(torch.sum(log_probs*probs,1),0) * self.beta
        vloss = torch.sum(adv.pow(2),0) / 2
        
        #vloss = F.smooth_l1_loss(v, rewards ) / 2
        loss = piloss + entropy + vloss
        
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
    