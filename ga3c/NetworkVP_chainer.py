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

import sys
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue
    
import re
import numpy as np
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import optimizer
from chainer import optimizers

from Config import Config

#import time


def init_like_torch(link):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
            l.b.data[:] = np.random.uniform(-stdv, stdv, size=l.b.data.shape)
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
            l.b.data[:] = np.random.uniform(-stdv, stdv, size=l.b.data.shape)
            


class a3c(chainer.ChainList):

    def __init__(self, n_input_channels=4, num_actions=32, num_gpu=0, rnn=0):
        self.num_gpu = num_gpu
        self.beta = Config.BETA_START
        self.log_epsilon = 1e-6
        outsize = 256
        bias = 0.1
        layers = [
            #L.Convolution2D(n_input_channels, 16, 8, stride=4, bias=bias,use_cudnn=1),
            #L.Convolution2D(16, 32, 4, stride=2, bias=bias,use_cudnn=1),
            #L.Linear(2592, outsize, bias=bias),
            L.Linear(16, outsize, bias=bias), #for CartPole-v0
            L.Linear(outsize, outsize, bias=bias),
            L.Linear(outsize, num_actions),
            L.Linear(outsize, 1),
        ]
        super(a3c, self).__init__(*layers)

        init_like_torch(self)
     

    @staticmethod
    def make_initial_state(batchsize, n_units, train=True):
        return {name: chainer.Variable(np.zeros((batchsize, n_units), dtype=np.float32), volatile=not train)
                for name in ('c1', 'h1')} 
        
        
    def predict_onestep(self, x, agent_idx):
        return Exception('blah')
        #h = chainer.Variable(x.reshape(x.shape[0],-1))
        #h1_in = F.relu(self[0](h))
        #c1, h1 = F.lstm(x['c1'], h1_in)
    
    @staticmethod   
    def loss(probs, v, y, a, log_epsilon=1e-6):
        probs = F.relu(probs - log_epsilon)
        log_probs = F.log(probs)
        adv = F.squeeze(y - v.data)    
  
        log_select_action_prob = F.sum( log_probs * a, axis=1 )
        piloss = -F.sum(log_select_action_prob * adv.data, axis=0) #we lose 10 ms here!
        entropy = F.sum(probs * log_probs, axis=1)
        entropy = F.sum(0.001 * entropy, axis=0)
        vloss = (v-y)**2
        vloss = F.squeeze( F.sum( vloss, axis=0 ) )  / 2
        loss = piloss + entropy + vloss
        #print(piloss.data, entropy.data, vloss.data)
        return loss
     
    def __call__(self, x, y, a):           
        #state = np.moveaxis(state, 3, 1)
        #h = chainer.Variable(state)
        
        h = chainer.Variable(x.reshape(x.shape[0],-1))
        
        y = y.astype(np.float32)
        y = y.reshape(y.shape[0],1)
        if self.num_gpu >= 0:
            h.to_gpu(self.num_gpu)          
            y = chainer.cuda.to_gpu(y,self.num_gpu) 
            a = chainer.cuda.to_gpu(a,self.num_gpu)
            

        for layer in self[:-2]:
            h = F.relu(layer(h))    
        
        probs, v = F.softmax(self[-2](h)),self[-1](h)
        
        #return a3c.piloss(probs,v,y,a)
        return a3c.loss(probs,v,y,a) #todo : make this much faster
        
    
    def pi_and_v(self, x, keep_same_state=False):
        #state = np.moveaxis(state, 3, 1)
        h = chainer.Variable(x.reshape(x.shape[0],-1))
        #h = chainer.Variable(state)
        if self.num_gpu >= 0:
            h.to_gpu(self.num_gpu)
        
        for layer in self[:-2]:
            h = F.relu(layer(h))
            
            
        h1_in = h + model.l1_h(state['h1'])
        c1, h1 = F.lstm(state['c1'], h1_in)
        h2_in = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state['h2'])
        c2, h2 = F.lstm(state['c2'], h2_in)
        y = model.l3(F.dropout(h2, train=train))
        
        
        probs, v = F.softmax(self[-2](h)),self[-1](h)
        return probs, v#, 
        #return cuda.to_cpu(probs.data),cuda.to_cpu(v.data)


#TODO : only thing we could do in TF side : maintain GPU data in GPU

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
        self.n_gpu = -1
        self.model = a3c(num_actions=num_actions,num_gpu=self.n_gpu)
          
        if self.n_gpu >= 0:
            self.model.to_gpu(self.n_gpu)

   
        #self.opt = optimizers.RMSprop(lr=1e-4,alpha=0.99,eps=1e-1)
        self.opt = optimizers.RMSpropGraves(lr=1e-4,alpha=0.99,eps=1e-1) #,momentum=0.0)
        #self.opt = optimizers.Adam(alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-8)
        
        self.opt.setup(self.model)
        #self.opt.add_hook(chainer.optimizer.GradientClipping(100.0))
        self.model.zerograds()
        
        self.t = 0

    def _create_graph(self):
        return Exception('not implemented')

    def reset_state(self, idx):
        return Exception('not implemented')
    

    def predict_p_and_v(self, x, rnn_state=None):  
        p, v, state = self.model.pi_and_v(x)
        
        return p.data, v.data #, infos
        #return p.data, v.data
    


    def train(self, x, y_r, a, trainer_id):        
        self.model(x, y_r, a).backward()   
        self.opt.update()
        self.model.zerograds()
    
    
    #just to check that we cannot train on offpolicy trajectories
    def train_offpolicy(self, x, y_r, a, trainer_id, episodes): 
        self.model.zerograds()
        #Gather probs
        loss = 0
        
        t = 0
        for episode in episodes:
            r = episode['r'].astype(np.float32)
            a_ = episode['a']
            i_ = episode['i']
            n = len(i_)
            pi, vi = [], []
            for f in i_:
                p,v = f
                pi.append(p)
                vi.append(v)
            p = F.concat(pi,axis=0)
            v = F.concat(vi,axis=0)
            loss += a3c.loss(p,v,r,a_)
            t += n
        
        #p = F.concat(pi, axis=0)
        #v = F.concat(vi, axis=0)
        #loss = a3c.loss(p,v,y_r.astype(np.float32), a)
        loss.backward()
        self.opt.update()


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
