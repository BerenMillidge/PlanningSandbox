
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import time
from RandomShooting import RandomShootingPlanner
from PI import PIPlanner
from CEM import CEMPlanner
from copy import deepcopy

from baselines.envs import TorchEnv, const

plan_horizon = 20
N_samples = 500
action_noise_sigma=1
top_candidates = 10
N_iterations = 1

env = TorchEnv("LunarLanderContinuous",200)
s = env.reset()
"""sorig = deepcopy(s)
for j in range(10):
    print("RESETTING")
    for i in range(50):
        a = env.action_space.sample()
        print(a)
        print(type(a))
        s,r,done = env.step(a)
        env.render()
    print("ORIGINAL s: ", sorig)
    print("state: before: ", s)
    env._env.set_state(sorig)
    s,r,done = env.step(np.array([0,0]))
    print("state after: ",s)
    print("state from env: ", env._env.lander.position, env._env.lander.linearVelocity,env._env.lander.angle, env._env.lander.angularVelocity)
    #bib"""


planner = RandomShootingPlanner(env, plan_horizon,N_samples, action_noise_sigma,discount_factor = 0.9)
#planner = PIPlanner(env, plan_horizon, N_samples, lambda_=5, noise_mu=0,noise_sigma=1)
#planner = CEMPlanner(env, plan_horizon, N_samples,top_candidates, N_iterations,discount_factor = 0.9)
for i in range(1000):
    a = planner(s)
    print("action: ",a)
    s,r,done= env.step(a)
    env.state = s
    print("reward: ",r)
    #_,_,_,_ = e.step(a)
    env.render()
    if done:
        #s = env.reset()
        s = env.reset()
        #e.state = s

    print(r)

#print(a)
