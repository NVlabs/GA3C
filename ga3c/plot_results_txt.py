# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:31:13 2017

@author: valeodevbox
"""

from __future__ import division
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=str, default='results.txt')
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, delimiter=', ')
    
    hour = scores['date']
    reward = scores['reward']
    
    time = []
    sh0 = hour[0].split(' ')[1].split(':')
    t0 = int(sh0[0]) + float(sh0[1])/60.0 + float(sh0[2])/3600.0
    for h in hour:
        sh = h.split(' ')[1].split(':')
        t = int(sh[0]) + float(sh[1])/60.0 + float(sh[2])/3600.0 - t0
        time.append(t)
    
    smoothing = 100
    reward_smooth = np.convolve(reward, np.ones((smoothing))/float(smoothing), 'same')
    
    xaxis = time
    plt.plot(xaxis, reward, color='b', alpha=0.2)
    plt.plot(xaxis, reward_smooth, color='b')
    
    plt.xlabel('hours')
    plt.ylabel('score')
    plt.legend(loc='best')
    fig_fname = args.scores + '.png'
    plt.savefig(fig_fname)
    plt.show()
    plt.pause(1)
    

if __name__ == '__main__':
    main()
