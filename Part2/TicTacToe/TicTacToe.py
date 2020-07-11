# -*- coding: utf-8 -*-
"""
Created on Mon June 15 00:02:42 2020

@author: Danish
"""
import numpy as np
from tkinter import Button, DISABLED
import tkinter.messagebox

# this class represents a tic-tac-toe game
# is a CS101-type of project
class Environment:
  def __init__(self, LENGTH=3):
    self.LENGTH = LENGTH
    self.board = np.zeros((LENGTH, LENGTH))
    self.x = -1 # represents an x on the board, player 1
    self.o = 1 # represents an o on the board, player 2
    self.winner = None
    self.ended = False
    self.num_states = 3**(LENGTH*LENGTH)

  def is_empty(self, i, j):
    return self.board[i,j] == 0

  def board_is_full(self):
      bord = np.nonzero(self.board.flatten())[0]
      return len(bord)==self.LENGTH**2

  def reward(self, sym):
    # no reward until game is over
    if not self.game_over():
      return 0

    # if we get here, game is over
    # sym will be self.x or self.o
    return 1 if self.winner == sym else 0

  def get_state(self):
    k = 0
    h = 0
    for i in range(self.LENGTH):
      for j in range(self.LENGTH):
        if self.board[i,j] == 0:
          v = 0
        elif self.board[i,j] == self.x:
          v = 1
        elif self.board[i,j] == self.o:
          v = 2
        h += (3**k) * v
        k += 1
    return h

  def game_over(self, force_recalculate=False):
    # returns true if game over (a player has won or it's a draw)
    # otherwise returns false
    # also sets 'winner' instance variable and 'ended' instance variable
    if not force_recalculate and self.ended:
      return self.ended
    
    # check rows
    for i in range(self.LENGTH):
      for player in (self.x, self.o):
        if self.board[i].sum() == player*self.LENGTH:
          self.winner = player
          self.ended = True
          return True

    # check columns
    for j in range(self.LENGTH):
      for player in (self.x, self.o):
        if self.board[:,j].sum() == player*self.LENGTH:
          self.winner = player
          self.ended = True
          return True

    # check diagonals
    for player in (self.x, self.o):
      # top-left -> bottom-right diagonal
      if self.board.trace() == player*self.LENGTH:
        self.winner = player
        self.ended = True
        return True
      # top-right -> bottom-left diagonal
      if np.fliplr(self.board).trace() == player*self.LENGTH:
        self.winner = player
        self.ended = True
        return True

    # check if draw
    if np.all((self.board == 0) == False):
      # winner stays None
      self.winner = None
      self.ended = True
      return True

    # game is not over
    self.winner = None
    return False

#  def is_draw(self):
#    return self.ended and self.winner is None


class Agent:
    def __init__(self, eps=0.1, alpha=0.5, LENGTH=3):
        self.eps = eps 
        self.alpha = alpha # learning rate
        self.LENGTH = LENGTH
        self.verbose = False
        self.state_history = []
  
    def setV(self, V):
        self.V = V

    def set_symbol(self, sym):
        self.sym = sym

    def set_verbose(self, v):
        # if true, will print values for each position on the board
        self.verbose = v

    def reset_history(self):
        self.state_history = []

    def take_action(self, env, ret=False):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        #best_state = None
        if r < self.eps:
          # take a random action
          if self.verbose:
            print("Taking a random action")
      
          possible_moves = []
          for i in range(self.LENGTH):
            for j in range(self.LENGTH):
              if env.is_empty(i, j):
                possible_moves.append((i, j))
          idx = np.random.choice(len(possible_moves))
          next_move = possible_moves[idx]
        else:
          # choose the best action based on current values of states
          # loop through all possible moves, get their values
          # keep track of the best value
          pos2value = {} # for debugging
          next_move = None
          best_value = -1
          for i in range(self.LENGTH):
            for j in range(self.LENGTH):
              if env.is_empty(i, j):
                # what is the state if we made this move?
                env.board[i,j] = self.sym
                state = env.get_state()
                env.board[i,j] = 0 # don't forget to change it back!
                pos2value[(i,j)] = self.V[state]
                if self.V[state] > best_value:
                  best_value = self.V[state]
                  #best_state = state
                  next_move = (i, j)
      
          # if verbose, draw the board w/ the values
          if self.verbose:
            print("Taking a greedy action")
            for i in range(self.LENGTH):
              print("------------------")
              for j in range(self.LENGTH):
                if env.is_empty(i, j):
                  # print the value
                  print(" %.2f|" % pos2value[(i,j)], end="")
                else:
                  print("  ", end="")
                  if env.board[i,j] == env.x:
                    print("x  |", end="")
                  elif env.board[i,j] == env.o:
                    print("o  |", end="")
                  else:
                    print("   |", end="")
              print("")
            print("------------------")
      
        # make the move
        env.board[next_move[0], next_move[1]] = self.sym
        #print(next_move)
        if ret:
            return next_move

    def update_state_history(self, s):
        self.state_history.append(s)

    def update(self, env):
        # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
        # where V(next_state) = reward if it's the most current state
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
          value = self.V[prev] + self.alpha*(target - self.V[prev])
          self.V[prev] = value
          target = value
        self.reset_history()



class BoardGUI():
    def __init__(self, tk, master=None):
        self.button1 = None; self.button2 = None; self.button3 = None
        self.button4 = None; self.button5 = None; self.button6 = None
        self.button7 = None; self.button8 = None; self.button9 = None
        self.tk=tk
    
    def disableButton(self):
        self.button1.configure(state=DISABLED)
        self.button2.configure(state=DISABLED)
        self.button3.configure(state=DISABLED)
        self.button4.configure(state=DISABLED)
        self.button5.configure(state=DISABLED)
        self.button6.configure(state=DISABLED)
        self.button7.configure(state=DISABLED)
        self.button8.configure(state=DISABLED)
        self.button9.configure(state=DISABLED)
        
    def set_symbol(self, sym):
        self.sym = sym
    
    def btnUpdate(self, next_move, text):
        i=next_move[0]; j=next_move[1]
        if i==0 and j==0:
            self.button1['text']=text
        elif i==0 and j==1:
            self.button2['text']=text
        elif i==0 and j==2:
            self.button3['text']=text
        elif i==1 and j==0:
            self.button4['text']=text
        elif i==1 and j==1:
            self.button5['text']=text
        elif i==1 and j==2:
            self.button6['text']=text
        elif i==2 and j==0:
            self.button7['text']=text
        elif i==2 and j==1:
            self.button8['text']=text
        elif i==2 and j==2:
            self.button9['text']=text
    
    def checkResults(self, p1, env):
        if env.game_over():
            if env.winner == p1.sym:
                self.disableButton()
                tkinter.messagebox.showinfo("Tic-Tac-Toe", 'AI Wins!')
            elif env.winner == None:
                self.disableButton()
                tkinter.messagebox.showinfo("Tic-Tac-Toe", 'It is a draw!')
            else:
                self.disableButton()
                tkinter.messagebox.showinfo("Tic-Tac-Toe", 'Congrats You have won the game!')
            if tkinter.messagebox.askyesno('Tic-Tac-Toe', 'Do you want to play again?'):
                self.play_game(p1, Environment())
            else:
                self.tk.destroy()
                
    def agents_turn(self, p1, env):
        ''' Agents Turn'''
        next_move = p1.take_action(env, ret=True)
        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        print(next_move)
        self.btnUpdate(next_move, text='X' if p1.sym==-1 else 'O')
        self.checkResults(p1, env)
        
    def btnClick(self, buttons, p1, env, loc):
        #print('Button at location {0} is clicked:'.format(loc))
        if buttons["text"] == " ":
            if env.is_empty(loc[0], loc[1]):
                env.board[loc[0], loc[1]] = self.sym
            print(env.board)
            buttons["text"] = "O" if self.sym==1 else "X"
            self.checkResults(p1, env)
            self.agents_turn(p1, env)
        else:
            tkinter.messagebox.showinfo("Tic-Tac-Toe", "Button already Clicked!")
            
    def play_game(self, p1, env):
        self.tk.title("Tic Tac Toe")
        self.button1 = Button(self.tk, text=" ", font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button1, p1, env, [0, 0]))
        self.button1.grid(row=0, column=0)
        
        self.button2 = Button(self.tk, text=' ', font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button2, p1, env, [0, 1]))
        self.button2.grid(row=0, column=1)
        
        self.button3 = Button(self.tk, text=' ',font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button3, p1, env, [0, 2]))
        self.button3.grid(row=0, column=2)
        
        self.button4 = Button(self.tk, text=' ', font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button4, p1, env, [1, 0]))
        self.button4.grid(row=1, column=0)
        
        self.button5 = Button(self.tk, text=' ', font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button5, p1, env, [1, 1]))
        self.button5.grid(row=1, column=1)
        
        self.button6 = Button(self.tk, text=' ', font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button6, p1, env, [1, 2]))
        self.button6.grid(row=1, column=2)
        
        self.button7 = Button(self.tk, text=' ', font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button7, p1, env, [2, 0]))
        self.button7.grid(row=2, column=0)
        
        self.button8 = Button(self.tk, text=' ', font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button8, p1, env, [2, 1]))
        self.button8.grid(row=2, column=1)
        
        self.button9 = Button(self.tk, text=' ', font='Times 20 bold', bg='gray', fg='white', height=4, 
                              width=8, command=lambda: self.btnClick(self.button9, p1, env, [2, 2]))
        self.button9.grid(row=2, column=2)
        if tkinter.messagebox.askyesno('Tic-Tac-Toe User Choice', 'Yes if AI should get first turn, No if you want to get first turn!'):
            self.agents_turn(p1, env)
        self.tk.mainloop()
        

class TicTacToe:
    def __init__(self, env):
        self.env = env

    def get_state_hash_and_winner(self, i=0, j=0):
        """ recursive function that will return all possible states (as ints) and who the corresponding winner is 
            for those states (if any) (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
            impossible games are ignored, i.e. 3x's and 3o's in a row simultaneously since that will never happen in a 
            real game """
        results = []
        
        for v in (0, self.env.x, self.env.o):
            self.env.board[i,j] = v # if empty board it should already be 0
            if j == 2:
                # j goes back to 0, increase i, unless i = 2, then we are done
                if i == 2:
                    # the board is full, collect results and return
                    state = self.env.get_state()
                    ended = self.env.game_over(force_recalculate=True)
                    winner = self.env.winner
                    results.append((state, winner, ended))
                else:
                    results += self.get_state_hash_and_winner(i + 1, 0)
            else:
                # increment j, i stays the same
                results += self.get_state_hash_and_winner(i, j + 1)
        return results
    
    
    def initialV_x(self, state_winner_triples):
      # initialize state values as follows
      # if x wins, V(s) = 1
      # if x loses or draw, V(s) = 0
      # otherwise, V(s) = 0.5
      V = np.zeros(self.env.num_states)
      for state, winner, ended in state_winner_triples:
        if ended:
          if winner == self.env.x:
            v = 1
          else:
            v = 0
        else:
          v = 0.5
        V[state] = v
      return V
    
    def initialV_o(self, state_winner_triples):
      # this is (almost) the opposite of initial V for player x
      # since everywhere where x wins (1), o loses (0)
      # but a draw is still 0 for o
      V = np.zeros(self.env.num_states)
      for state, winner, ended in state_winner_triples:
        if ended:
          if winner == self.env.o:
            v = 1
          else:
            v = 0
        else:
          v = 0.5
        V[state] = v
      return V
    
    def play_agents(self, p1, p2, env):
      # loops until the game is over
      current_player = None
      while not env.game_over():
        # alternate between players
        # p1 always starts first
        if current_player == p1:
          current_player = p2
        else:
          current_player = p1
        # current player makes a move
        current_player.take_action(env)
        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
      
      # do the value function update
      p1.update(env)
      p2.update(env)