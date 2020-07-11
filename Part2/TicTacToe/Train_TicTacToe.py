# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:36:47 2020

@author: Danish
"""
from TicTacToe import Environment, Agent, TicTacToe, BoardGUI
from tqdm import tqdm
import pickle
from tkinter import Tk

def train_agent(T=10000, save=True, ret=False, name=None):
    if save:
        if type(name) != str:
            raise TypeError('Provide proper string type value to the argument name.')
    # train the agent
    p1 = Agent()
    p2 = Agent()
    
    # set initial V for p1 and p2
    env = Environment()
    #Create the object for TicTacToe class 
    tt = TicTacToe(env)
    state_winner_triples = tt.get_state_hash_and_winner()
    
    
    Vx = tt.initialV_x(state_winner_triples)
    p1.setV(Vx)
    Vo = tt.initialV_o(state_winner_triples)
    p2.setV(Vo)
    
    # give each player their symbol
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)
    
    for t in tqdm(range(T)):
      tt.play_agents(p1, p2, Environment())
    if save:
        with open(name, 'wb') as f:
            pickle.dump(p1, f)
    if ret:
        p1, p2
        
def play_TicTacToe(name):
    tk = Tk()
    with open(name, 'rb') as f:
        p1 = pickle.load(f)
    env = Environment()
    bg = BoardGUI(tk)
    bg.set_symbol(env.o)
    bg.play_game(p1, Environment())
    
if __name__=='__main__':
    #train_agent(T=10000, save=True, name='AgentAI')
    play_TicTacToe(name='AgentAI')
