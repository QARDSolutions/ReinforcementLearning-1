# -*- coding: utf-8 -*-
"""
Created on Sat May 23 02:17:13 2020

@author: Danish
"""


import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import simulate_epsilon

class MAB:
    def __init__(self, m):
        self.m=m
        self.mean=0
        self.N=0
    def pull_arm(self):
        return np.random.randn()+self.m
    def update_mean(self, x):
        self.N += 1
        self.mean = (1-1/self.N)*self.mean+(1.0/self.N)*x
   
def ucb(mean, n, nj):
    if nj==0:
        return float('inf')   
    return mean + np.sqrt(2*np.log(n)/nj)      
   
def simulate_ucb(means, N):
    mabs = [MAB(means[0]), MAB(means[1]), MAB(means[2])]
    
    samples = np.zeros(N)
    for i in range(N):
        lst = []
        for b in mabs:
            lst.append(ucb(b.mean, i+1, b.N))
        idx = np.argmax(lst)
        x = mabs[idx].pull_arm()
        mabs[idx].update_mean(x)
        samples[i] = x
    cum_avg = np.cumsum(samples)/np.arange(1,N+1)
    #print('For Epsilon Value: {0}, Number of Explored samples: {1}/{2}'.format(epsilon, len(explore), N))
    #plotting
    plt.plot(cum_avg, label='Cumulative Average-UCB')
    plt.plot(np.ones(N)*means[0])
    plt.plot(np.ones(N)*means[1])
    plt.plot(np.ones(N)*means[2])
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.title('Moving Average Plot - Log Scale, UCB')
    plt.ylabel('Mean Rate of choosing best arm')
    plt.xlabel('Time Step')
    plt.show()

    return cum_avg   


means = [1, 2 , 3]
N = int(10e4)
cum_avg1 = simulate_epsilon(means, N, epsilon=0.1)
oiv = simulate_ucb(means, N)

plt.plot(cum_avg1, label='epsilon = 0.1')
plt.plot(oiv, label = 'Upper Confidence Bound')
plt.legend(loc='lower right')
plt.xscale('log')
plt.title('Moving Average Plot - Logf Scale')
plt.ylabel('Mean Rate of choosing best arm')
plt.xlabel('Time Step')
plt.show()
