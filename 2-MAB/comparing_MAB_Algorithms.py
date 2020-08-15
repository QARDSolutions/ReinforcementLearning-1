# -*- coding: utf-8 -*-
"""
Created on Sun May 24 03:36:18 2020

@author: Danish
"""


import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import MAB, simulate_epsilon
from OIV import simulate_oiv
from UCB import simulate_ucb
from BayesianSampling import simulate_bayesian

def simulate_decaying_epsilon(means, N):
    mabs = [MAB(means[0]), MAB(means[1]), MAB(means[2])]
    
    samples = np.zeros(N)
    explore = []
    for i in range(N):
        rand = np.random.random()
        if rand<1/(i+1):
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
    #plotting
    plt.plot(cum_avg, label='Cumulative Average- Decaying Epsilon')
    plt.plot(np.ones(N)*means[0])
    plt.plot(np.ones(N)*means[1])
    plt.plot(np.ones(N)*means[2])
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.title('Moving Average Plot - Log Scale, Decaying Epsilon')
    plt.ylabel('Mean Rate of choosing best arm')
    plt.xlabel('Time Step')
    plt.show()
    return cum_avg 

if __name__=='__main__':
    means = [1, 2 , 3]
    N = int(10e4)
    ep_g = simulate_epsilon(means, N, epsilon=0.1)
    dec_ep = simulate_decaying_epsilon(means, N)
    oiv = simulate_oiv(means, N, bound=10)
    ucb_res = simulate_ucb(means, N)
    bayes = simulate_bayesian(means, N)

    
    plt.plot(ep_g, label='Epsilon = 0.1')
    plt.plot(dec_ep, label='Decaying Epsilon')
    plt.plot(oiv, label = 'Optmistic Initail Values')
    plt.plot(ucb_res, label = 'Upper Confidence Bound')
    plt.plot(bayes, label = 'Bayesian Sampling')
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.title('Moving Average Plot - Log Scale')
    plt.ylabel('Mean Rate of choosing best arm')
    plt.xlabel('Time Step')
    plt.show()
