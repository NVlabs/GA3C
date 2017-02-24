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
import numpy as np
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import optimizer

from Config import Config

import time

class RMSpropAsync(optimizer.GradientMethod):

    """RMSprop for asynchronous methods.
    
    The only difference from chainer.optimizers.RMSprop in that the epsilon is
    outside the square root."""

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        state['ms'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        ms = state['ms']
        grad = param.grad

        ms *= self.alpha
        ms += (1 - self.alpha) * grad * grad
        param.data -= self.lr * grad / numpy.sqrt(ms + self.eps)

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / sqrt(ms + eps);''',
            'rmsprop')(param.grad, self.lr, self.alpha, self.eps,
                       param.data, state['ms'])
            
class NonbiasWeightDecay(object):

    """Optimizer hook function for weight decay regularization.

    """
    name = 'NonbiasWeightDecay'

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, opt):
        if cuda.available:
            kernel = cuda.elementwise(
                'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')

        rate = self.rate
        for name, param in opt.target.namedparams():
            if name == 'b' or name.endswith('/b'):
                continue
            p, g = param.data, param.grad
            with cuda.get_device(p) as dev:
                if int(dev) == -1:
                    g += rate * p
                else:
                    kernel(p, rate, g)


def init_like_torch(link):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape)
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape)


class a3c(chainer.ChainList):

    def __init__(self, n_input_channels=4, num_actions=32, num_gpu=0, rnn=0, beta=1e-2):
        self.num_gpu = num_gpu
        self.beta = beta
        
        outsize = 256
        bias = 0.1
        self.activation = F.leaky_relu
        layers = [
            L.Convolution2D(n_input_channels, 16, 8, stride=4, bias=bias,use_cudnn=1),
            L.Convolution2D(16, 32, 4, stride=2, bias=bias,use_cudnn=1),
            L.Linear(2592, outsize, bias=bias),
            L.Linear(outsize, num_actions),
            L.Linear(outsize, 1),
        ]
        super(a3c, self).__init__(*layers)
        
        init_like_torch(self)
        
    def __call__(self, state, y, action_index):           
        state = np.moveaxis(state, 3, 1)
        h = chainer.Variable(state)
        if self.num_gpu >= 0:
            h.to_gpu(self.num_gpu) 
            y = y.reshape(y.shape[0],1)
            y = chainer.cuda.to_gpu(y,self.num_gpu) 
            a = chainer.cuda.to_gpu(action_index,self.num_gpu)

        for layer in self[:-2]:
            h = self.activation(layer(h))    
        probs, v = F.softmax(self[-2](h)),self[-1](h)
        log_probs = F.log(probs)

        log_select_action_prob = F.sum( log_probs * a, axis=1 )
        adv = F.reshape((y - v.data), log_select_action_prob.data.shape)          
        piloss = -F.sum(log_select_action_prob * adv, axis=0) #we lose 10 ms here!
        entropy = F.sum(probs * log_probs, axis=1)
        entropy = F.sum(self.beta * entropy, axis=0)
        vloss = (v-y)**2
        vloss = F.reshape( F.sum( vloss, axis=0 ) , ())  / 4
        loss = piloss + entropy
        return loss

    def pi_and_v(self, state, keep_same_state=False):
        state = np.moveaxis(state, 3, 1)
        h = chainer.Variable(state)
        if self.num_gpu >= 0:
            h.to_gpu(self.num_gpu)
        
        for layer in self[:-2]:
            h = self.activation(layer(h))
            
        probs, v = F.softmax(self[-2](h)),self[-1](h)
        return cuda.to_cpu(probs.data),cuda.to_cpu(v.data)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()


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
        self.n_gpu = 0
        self.model = a3c(num_actions=num_actions,num_gpu=self.n_gpu)
          
        if self.n_gpu >= 0:
            self.model.to_gpu(self.n_gpu)
        
        
        self.opt = RMSpropAsync(lr=1e-4,eps=1e-1,alpha=0.99)      
        self.opt.setup(self.model)
        self.opt.add_hook(chainer.optimizer.GradientClipping(40.0))
        #self.opt.add_hook(NonbiasWeightDecay(1e-5))

    def _create_graph(self):
        return Exception('not implemented')

    def reset_state(self, idx):
        return Exception('not implemented')


    def predict_p_and_v(self, x):
        p, v = self.model.pi_and_v(x)
        return p, v
    
    def train(self, x, y_r, a, trainer_id):        
        self.model.zerograds()
        
        st = time.time()
        self.model(x, y_r, a).backward()
         
        #norm = self.opt.compute_grads_norm()
        self.opt.update(  )
        print( 'total :',(time.time()-st)*1000, ' ms')
        

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
