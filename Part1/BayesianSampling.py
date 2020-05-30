# -*- coding: utf-8 -*-
"""
Created on Sun May 24 02:56:54 2020

@author: Danish
"""


import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import simulate_epsilon

class MAB:
    def __init__(self, m):
        self.m=m
        self.predicted_mean = 0
        self.lambda_ = 1
        self.sum_x = 0
        self.tau = 1
    def pull_arm(self):
        return np.random.randn()+self.m
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean
    def update_mean(self, x):
        lambda0 = self.lambda_
        self.lambda_ += self.tau
        self.sum_x += x
        self.predicted_mean = (self.predicted_mean*lambda0 +self.tau*self.sum_x)/self.lambda_
   
def ucb(mean, n, nj):
    if nj==0:
        return float('inf')   
    return mean + np.sqrt(2*np.log(n)/nj)      
   
def simulate_bayesian(means, N):
    mabs = [MAB(means[0]), MAB(means[1]), MAB(means[2])]
    
    samples = np.zeros(N)
    for i in range(N):
        lst = []
        for b in mabs:
            lst.append(b.sample())
        idx = np.argmax(lst)
        x = mabs[idx].pull_arm()
        mabs[idx].update_mean(x)
        samples[i] = x
    cum_avg = np.cumsum(samples)/np.arange(1,N+1)
    #plotting
    plt.plot(cum_avg, label='Cumulative Average- Bayesian Sampling')
    plt.plot(np.ones(N)*means[0])
    plt.plot(np.ones(N)*means[1])
    plt.plot(np.ones(N)*means[2])
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.title('Moving Average Plot - Log Scale, Bayesian Sampling')
    plt.ylabel('Mean Rate of choosing best arm')
    plt.xlabel('Time Step')
    plt.show()

    return cum_avg   

if __name__=='__main__':
    means = [1, 2 , 3]
    N = int(10e4)
    ep_g = simulate_epsilon(means, N, epsilon=0.1)
    bayes = simulate_bayesian(means, N)
    
    plt.plot(ep_g, label='Epsilon = 0.1')
    plt.plot(bayes, label = 'Bayesian Sampling')
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.title('Moving Average Plot - Log Scale')
    plt.ylabel('Mean Rate of choosing best arm')
    plt.xlabel('Time Step')
    plt.show()
