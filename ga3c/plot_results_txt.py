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

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)  

def prepare_time_axis(hour):
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
        
    return time

def addplot(filename,ax,color,label):
    scores = pd.read_csv(filename, delimiter=', ')
    hour = scores['date']
    reward = scores['reward']  
    time = prepare_time_axis(hour)

    mean_window = 100
    std_window = 200
    r_std = reward.rolling(window = std_window).std()
    r = reward.rolling(window = mean_window).mean()

    ax.plot(time, r, color=color,label=label)
    ax.fill_between(time, r-r_std, r+r_std, color=color, alpha=0.2, linewidth=0)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=str, default='results.txt')
    args = parser.parse_args()
    
    fig, axarr = plt.subplots(1, sharex=True, figsize=(8, 8))
    
    scores1 = 'results_ff.txt'
    scores2 = 'results.txt'

    addplot(scores1, axarr,tableau20[0],'batch16.ff')
    addplot(scores2, axarr,tableau20[4],'batch16.lstm')

    plt.xlabel('hours')
    plt.ylabel('CartPole-V0.score')
    plt.legend(loc='best')
    fig_fname = args.scores + '.png'
    plt.savefig(fig_fname)
    plt.show()
    plt.pause(1)
    

if __name__ == '__main__':
    main()
