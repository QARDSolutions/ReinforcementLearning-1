# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:39:58 2020

@author: Danish
"""

########## Explore-Exploit Dilemma using Epsilon Greedy Approach ##########

import numpy as np
import matplotlib.pyplot as plt

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
        
def simulate_epsilon(means, N, epsilon):
    mabs = [MAB(means[0]), MAB(means[1]), MAB(means[2])]
    
    samples = np.zeros(N)
    explore = []
    for i in range(N):
        rand = np.random.random()
        if rand<epsilon:
            idx = np.random.choice(3)
            explore.append(i)
        else:
            lst = []
            for b in mabs:
                lst.append(b.mean)
            idx = np.argmax(lst)
        x = mabs[idx].pull_arm()
        mabs[idx].update_mean(x)
        samples[i] = x
    cum_avg = np.cumsum(samples)/np.arange(1,N+1)
    print('For Epsilon Value: {0}, Number of Explored samples: {1}/{2}'.format(epsilon, len(explore), N))
    #plotting
    plt.plot(cum_avg, label='Cumulative Average-Epsilon Greedy')
    plt.plot(np.ones(N)*means[0])
    plt.plot(np.ones(N)*means[1])
    plt.plot(np.ones(N)*means[2])
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.title('Moving Average Plot - Logf Scale, epsilon: {0}'.format(epsilon))
    plt.ylabel('Mean Rate of choosing best arm')
    plt.xlabel('Time Step')
    plt.show()

    return cum_avg   

means = [1, 2 , 3]
N = int(10e4)
cum_avg1 = simulate_epsilon(means, N, epsilon=0.1)
cum_avg05 = simulate_epsilon(means, N, epsilon=0.05)
cum_avg01 = simulate_epsilon(means, N, epsilon=0.01)

plt.plot(cum_avg1, label='epsilon = 0.1')
plt.plot(cum_avg05, label='epsilon = 0.05')
plt.plot(cum_avg01, label='epsilon = 0.01')
plt.legend(loc='lower right')
plt.xscale('log')
plt.title('Moving Average Plot - Logf Scale')
plt.ylabel('Mean Rate of choosing best arm')
plt.xlabel('Time Step')
plt.show()


