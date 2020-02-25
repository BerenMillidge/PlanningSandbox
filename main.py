# the goal of this is just to test how well planners do on various environments when they have access to the true environmental states
#ultimately I want to turn this into a general sandbox for experimenting an visualizing with different continuous action planners
#so I can really understand the differences in the algorithms and potentially make improvements which would be good
#also will end up as a repo of implementations of different algorithms
#first test is just to get the baselines sorted out so I can compare PI vs CEM on a variety of tasks given the real environment and rewards
#then I can plan my own versions and improvements

import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import time
from RandomShooting import RandomShootingPlanner
from PI import PIPlanner
from CEM import CEMPlanner

from baselines.envs import TorchEnv, const

#serously, getting a good mountain car planning algorithm to work is HARD!
# it is majorly sparse!

plan_horizon = 20
N_samples = 100
action_noise_sigma=1

env = TorchEnv("pendulum",200)
e = gym.make("Pendulum-v0")
s = env.reset()
s = e.reset()
#planner = RandomShootingPlanner(env, plan_horizon,N_samples, action_noise_sigma,discount_factor = 0.9)
planner = PIPlanner(env, plan_horizon, N_samples, lambda_=5, noise_mu=0,noise_sigma=1)
for i in range(1000):
    a = planner(s)
    print("action: ",a)
    s,r,done,_ = e.step(a)
    e.state = s
    print("reward: ",r)
    #_,_,_,_ = e.step(a)
    e.render()
    if done:
        #s = env.reset()
        s = e.reset()
        #e.state = s

    print(r)

#print(a)
