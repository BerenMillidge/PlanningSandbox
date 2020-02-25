
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import time
from RandomShooting import RandomShootingPlanner
from PI import PIPlanner
from CEM import CEMPlanner
from copy import deepcopy
import sys
import argparse
import subprocess
from datetime import datetime

from baselines.envs import TorchEnv, const

def main(args):
    env = TorchEnv(args.env_name,args.max_episode_len)
    s = env.reset()
    if args.planner_type == "RandomShooting":
        planner = RandomShootingPlanner(env,
         plan_horizon=args.plan_horizon,
         N_samples=args.N_samples,
         action_noise_sigma=args.action_noise_sigma,
         discount_factor = args.discount_factor)
    elif args.planner_type == "PI":
        planner = PIPlanner(env,
        plan_horizon=args.plan_horizon,
        N_samples=args.N_samples,
        lambda_=args.PI_lambda_,
        noise_mu=0,
        noise_sigma=args.action_noise_sigma)
    elif args.planner_type == "CEM":
        planner = CEMPlanner(env,
        plan_horizon=args.plan_horizon,
        n_candidates = args.N_samples,
        top_candidates = args.CEM_top_candidates,
        optimisation_iters=args.CEM_iterations,
        action_noise_sigma = args.action_noise_sigma,
        discount_factor = args.discount_factor)

    results = np.zeros([args.N_episodes, args.max_episode_len])
    for i_ep in range(args.N_episodes):
        s = env.reset()
        for j in range(args.max_episode_len):
            a = planner(s)
            #print("action: ",a)
            s,r,done= env.step(a)
            print("reward: ",r)
            results[i_ep, j] = r
            #env.render()
            if done:
                s = env.reset()

        np.save(args.logdir, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner_type", type=str)
    parser.add_argument("--plan_horizon", type=int)
    parser.add_argument("--N_samples",type=int)
    parser.add_argument("--action_noise_sigma",type=float, default=1.0)
    parser.add_argument("--CEM_iterations", type=int, default=5)
    parser.add_argument("--CEM_top_candidates", type=int, default=50)
    parser.add_argument("--PI_lambda_",type=float, default=1)
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--max_episode_len", type=int, default=2)
    parser.add_argument("--N_episodes", type=int, default=1)
    parser.add_argument("--env_name", type=str)

    args = parser.parse_args()
    main(args)
