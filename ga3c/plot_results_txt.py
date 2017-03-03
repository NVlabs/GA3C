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
    tprev = t0
    total_time = 0
    dt = 0
    for h in hour:
        sh = h.split(' ')[1].split(':')  
        tt = int(sh[0]) + float(sh[1])/60.0 + float(sh[2])/3600.0
        if tt < tprev: 
            dt = total_time
            t0 = tt
        elif (tt-tprev) > 0.5:
            dt = total_time
            t0 = tt
        else:
            total_time += (tt-tprev)
        t = tt - t0 + dt
        time.append(t)
        tprev = tt
    
    smoothing = 100
    reward_smooth = np.convolve(reward, np.ones((smoothing))/float(smoothing), 'same')
    
    xaxis = time
    plt.plot(xaxis, reward, color='b', alpha=0.2)
    plt.plot(xaxis, reward_smooth, label='reward', color='b')
    
    plt.xlabel('hours')
    plt.ylabel('score')
    plt.legend(loc='best')
    fig_fname = args.scores + '.png'
    plt.savefig(fig_fname)
    plt.show()
    plt.pause(1)
    

if __name__ == '__main__':
    main()
