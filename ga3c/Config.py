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

import gym

#register your own environment
gym.envs.register(
    id='PongDeterministic-v100',
    entry_point='gym.envs.atari:AtariEnv',
    reward_threshold=21.0,
    kwargs={'game':'pong', 'obs_type':  'image', 'frameskip':4, 'repeat_action_probability': 0.25},
    nondeterministic=False
)

class Config:

    #########################################################################
    # Game configuration

    # Name of the game, with version (e.g. PongDeterministic-v0)
    ATARI_GAME = 'PongDeterministic-v100'

    # Enable to see the trained agent in action
    PLAY_MODE = False
    # Enable to train
    TRAIN_MODELS = True
    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = False
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 0 

    #########################################################################
    # Number of agents, predictors, trainers and other system settings
    
    # If the dynamic configuration is on, these are the initial values.
    # Number of Agents
    AGENTS = 32 
    # Number of Predictors
    PREDICTORS = 2
    # Number of Trainers
    TRAINERS = 2

    # Device
    DEVICE = 'gpu:0'

    # Enable the dynamic adjustment (+ waiting time to start it)
    DYNAMIC_SETTINGS = True
    DYNAMIC_SETTINGS_STEP_WAIT = 20
    DYNAMIC_SETTINGS_INITIAL_WAIT = 10

    #########################################################################
    # Algorithm parameters

    # Discount factor
    DISCOUNT = 0.99
    
    # Tmax
    TIME_MAX = 20
    
    # Reward Clipping
    REWARD_MIN = -1
    REWARD_MAX = 1

    # Max size of the queue
    MAX_QUEUE_SIZE = 20
    PREDICTION_BATCH_SIZE = 128

    # Input of the DNN
    STACKED_FRAMES = 1
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84

    # Total number of episodes and annealing frequency
    EPISODES = 4000
    ANNEALING_EPISODE_COUNT = 4000

    # Entropy regualrization hyper-parameter
    BETA_START = 0.01
    BETA_END = 0.01

    # Learning rate
    LEARNING_RATE_START = 0.001
    LEARNING_RATE_END = 0.001

    # RMSProp parameters
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1

    # Dual RMSProp - we found that using a single RMSProp for the two cost function works better and faster
    DUAL_RMSPROP = False
    
    # Gradient clipping
    USE_GRAD_CLIP = True
    GRAD_CLIP_NORM = 40.0 
    # Epsilon (regularize policy lag in GA3C)
    LOG_EPSILON = 1e-6
    # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower
    TRAINING_MIN_BATCH_SIZE = 20
    
    # USE RNN - can help to converge but current version is much slower than FF
    USE_RNN = True
    NCELLS = 256
    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = False
    # Update TensorBoard every X training steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 1000
    
    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 1
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 1000

    # Results filename
    RESULTS_FILENAME = 'results.txt'
    # Network checkpoint name
    NETWORK_NAME = 'network'

    #########################################################################
    # More experimental parameters here
    
    # Minimum policy
    MIN_POLICY = 0.0
    # Use log_softmax() instead of log(softmax())
    USE_LOG_SOFTMAX = True
