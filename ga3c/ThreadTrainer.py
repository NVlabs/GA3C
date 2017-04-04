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

from threading import Thread
import numpy as np
from Config import Config
from replay_buffer import ReplayBuffer

class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False
        self.maxbufsize = 10000 / Config.TIME_MAX
        self.replay_buf = ReplayBuffer(self.maxbufsize)
        
        
    @staticmethod
    def _dynamic_pad(x_,r_,a_):
        t = x_.shape[0]
        if t != Config.TIME_MAX and Config.USE_RNN:
            xt = np.zeros((Config.TIME_MAX, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES),dtype=np.float32)
            rt = np.zeros((Config.TIME_MAX,1),dtype=np.float32)
            at = np.zeros((Config.TIME_MAX, a_.shape[1]),dtype=np.float32)
            xt[:t] = x_; rt[:t] = r_; at[:t] = a_
            x_ = xt; r_ = rt; a_ = at;
        return x_, r_, a_, t 
                    
    def run(self):
        while not self.exit_flag:
            batch_size = 0
            lengths = []
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                idx, x_, r_, a_, c_, h_ = self.server.training_q.get()
                
                x_,r_,a_,t = ThreadTrainer._dynamic_pad(x_,r_,a_)
                lengths.append(t)
                
                if batch_size == 0:
                    x__ = x_; r__ = r_; a__ = a_; c__ = c_; h__ = h_; 
                else:
                    #x__, r__, a__, c__, h__ = ThreadTrainer._concat((x__, r__, a__, c__, h__),(x_, r_, a_, c_, h_))
                    x__ = np.concatenate((x__, x_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))
                    c__ = np.concatenate((c__, c_))
                    h__ = np.concatenate((h__, h_))
                
                batch_size += x_.shape[0] #we change meaning of batch
            
            if Config.TRAIN_MODELS:
                self.server.train_model(x__, r__, a__,c__,h__, lengths) 


    @staticmethod
    def _concat(exp1,exp2):
        x1,r1,a1,c1,h1 = exp1
        x2,r2,a2,c2,h2 = exp2
        x1 = np.concatenate((x1, x2))
        r1 = np.concatenate((r1, r2))
        a1 = np.concatenate((a1, a2))
        c1 = np.concatenate((c1, c2))
        h1 = np.concatenate((h1, h2))
        return x2,r2,a2,c2,h2
        
    
    def get_batch_replay(self):  
        lengths = []
        batches = self.replay_buf.get_batch(4)
        x__, r__, a__, c__, h__, l = batches[0]
        lengths.append(l)
        for i in range(1,4):
            x_, r_, a_, c_, h_, l =batches[i]
            x__ = np.concatenate((x__, x_))
            r__ = np.concatenate((r__, r_))
            a__ = np.concatenate((a__, a_))
            c__ = np.concatenate((c__, c_))
            h__ = np.concatenate((h__, h_))
            lengths.append(l)
        return x__, r__, a__, c__, h__, l 
        
        
    #mix on & off policy?
    def run_offpolicy(self):
        batch_size = 0
        lengths = []
        while not self.exit_flag:
            idx, x_, r_, a_, c_, h_ = self.server.training_q.get()
            
            x_,r_,a_,t = ThreadTrainer._dynamic_pad(x_,r_,a_)
            if batch_size == 0:
                x__ = x_; r__ = r_; a__ = a_; c__ = c_; h__ = h_; 
            else:
                x__ = np.concatenate((x__, x_))
                r__ = np.concatenate((r__, r_))
                a__ = np.concatenate((a__, a_))
                c__ = np.concatenate((c__, c_))
                h__ = np.concatenate((h__, h_))
                
            if batch_size >= Config.TRAINING_MIN_BATCH_SIZE:
                self.server.train_model(x__, r__, a__,c__,h__, lengths)
                batch_size = 0
                lengths = []
                
            self.replay_buf.add(x_,r_,a_,c_,h_, t)        
            if self.replay_buf.count() >= self.maxbufsize / 4:
                x, r, a, c, h, l = self.get_batch_replay()              
                if Config.TRAIN_MODELS:
                    self.server.train_model_off(x, r, a, c, h, l) 
                 