'''
This file is use to train the last rules we added, namely :
- leftward_with_another_color: as the name implies, another cue but the same behavior
- leftward_closest_endpoints: leftward, but instead of drawing consistently left to right we draw to the closest endpoint
- closest_from_left: closest barycenter, but always draw left to right. 

All of these rules are not very interesting by themselves, more so to confirm we understand the otehr cases.
'''


import torch as tch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import pandas as pd

from train import train
from copy import deepcopy
from agent import SaccadeAgent
from env import RuleBoards
from itertools import product
from oracle import RuleOracle, RandomAgent
import os
import sys
from tqdm import tqdm

from test import test_suite

# Technical bother with local vs cluster paths...
wdir = '/scratch/atf6569/saccade_drawing/'
OUT_DIR = wdir + 'full_coverage_rules/'   


args = {
        'root_output_folder': OUT_DIR,
        'run_name': 'paired_variants',

        'board_params': {
            'n_envs': 64,
            'n_symbols_min': 4,
            'n_symbols_max': 4,
            'reward_type': 'default',
            'reward_params':  {'overlap_criterion': .2},
            'ordering_sensitivity': 0, # in pixels
            'all_envs_start_identical': False, 
            'timeout': None,
            'mirror': False,

            'allowed_symbols': ['line_0', 'line_1', 'line_2', 'line_3'],
            'allowed_rules': ['closest', 'leftward', 'leftward_with_another_color', 'leftward_closest_endpoints', 'closest_from_left'],
        },

        'oracle_params': {
            'sensitivity': .1,
            'noise': .02,
        },

        'agent_params': {
            'fovea_ablated': False,
            'no_saccade_loss': True,


            # 'peripheral_net_params': {
            #     'n_pixels_in': 128,
            #     'cnn_n_featuremaps': [128, 128, 64, 64, 32],
            #     'cnn_kernel_sizes': [5, 5, 5, 3, 3],
            #     'cnn_kernel_strides': [1, 1, 1, 1, 1],
            #     'fc_sizes': [512, 512],
            #     'rnn_size': 512,
            #     'recurrence_type': 'rnn',
            #     'recurrent_steps': 1, 
            #     },

            # Try with smaller network, see if anything interesting happens
            'peripheral_net_params': {
                'n_pixels_in': 128,
                'cnn_n_featuremaps': [64, 64, 64, 64, 32],
                'cnn_kernel_sizes': [5, 5, 5, 3, 3],
                'cnn_kernel_strides': [1, 1, 1, 1, 1],
                'fc_sizes': [512, 512],
                'rnn_size': 512,
                'recurrence_type': 'rnn',
                'recurrent_steps': 1, 
                },

                
            'foveal_net_params': {
                'n_pixels_in': 32,
                'cnn_n_featuremaps': [128, 64, 64],
                'cnn_kernel_sizes': [5, 5, 3,],
                'cnn_kernel_strides': [1, 1, 1,],
                'fc_sizes': [512, 512],
                'rnn_size': 512,
                'recurrence_type': 'rnn',
                'recurrent_steps': 1, 
                },
        },

        'buffer_params': {
            'buffer_size': 5_000,
        },

        'training_params': {
            'n_updates': 40_000,
            'test_every': 1000,
            'lr': 5e-4,
            'batch_size': 128,
            # How many env steps we do collecting data before updating the agent
            'rollout_length': 20,
            # How many batches / GD steps we do after each rollout phase 
            'gd_steps_per_rollout': 5, # about rollout steps * n_envs / batch_size
            'start_agent_only_after': 500, # This is kind of a weird crutch, try to get rid of it !
        },
    }



def run_one_seed_training(seed):
    local_args = deepcopy(args)
    local_args['seed'] = seed
    train(local_args)




def run_one_seed_testing(seed, n_steps_plot=10, n_steps_total=100):
    # Still running !
    agent = SaccadeAgent(args['agent_params']['peripheral_net_params'], args['agent_params']['foveal_net_params'])
    # Not completely done running, but still try it out
    # agent.load_state_dict(tch.load(OUT_DIR + f'all_rules/default/seed{seed}/final_agent.pt'))
    agent.load_state_dict(tch.load(OUT_DIR + f'paired_variants/seed{seed}/agent.pt'))

    oracle = RuleOracle(**args['oracle_params']) # RuleOracle does not care for number of lines, it reads it from the environment
    random_agent = RandomAgent(amplitude=.5, seed=seed)

    # # This is just a sanity check to ensure we are not using the same network for frankensteining, which would not be very interesting
    # assert not tch.allclose(four_lines_agent.peripheral_net.convnet[0].weight, one_to_four_agent.peripheral_net.convnet[0].weight)

    ref_board_params = deepcopy(args['board_params'])

    only_two_board_params = deepcopy(args['board_params'])
    only_two_board_params['seed'] = seed
    only_two_board_params['n_symbols_min'] = 2
    only_two_board_params['n_symbols_max'] = 2
    only_two_env = RuleBoards(only_two_board_params)

    only_three_board_params = deepcopy(args['board_params'])
    only_three_board_params['seed'] = seed
    only_three_board_params['n_symbols_min'] = 3
    only_three_board_params['n_symbols_max'] = 3
    only_three_env = RuleBoards(only_three_board_params)

    only_four_env = RuleBoards(args['board_params'])

    five_to_six_board_params = deepcopy(args['board_params'])
    five_to_six_board_params['seed'] = seed
    five_to_six_board_params['n_symbols_min'] = 5
    five_to_six_board_params['n_symbols_max'] = 6

    mirrored_five_to_six_board_params = deepcopy(args['board_params'])
    mirrored_five_to_six_board_params['seed'] = seed
    mirrored_five_to_six_board_params['n_symbols_min'] = 5
    mirrored_five_to_six_board_params['n_symbols_max'] = 6
    mirrored_five_to_six_board_params['mirrored'] = True

    five_to_six_env = RuleBoards(five_to_six_board_params)
    mirrored_five_to_six_env = RuleBoards(mirrored_five_to_six_board_params)

    # This one will let us see if giving it more time helps it get over such a big generalization leap
    # NOTE: 8 is a bit much, a lot of them get stuck; keep it just to show we're not hiding anything
    eight_timeout_thirty_board_params = deepcopy(args['board_params'])
    eight_timeout_thirty_board_params['seed'] = seed
    eight_timeout_thirty_board_params['n_symbols_min'] = 8
    eight_timeout_thirty_board_params['n_symbols_max'] = 8
    eight_timeout_thirty_board_params['timeout'] = 30
    eight_timeout_thirty_env = RuleBoards(eight_timeout_thirty_board_params)

    six_mirrored_timeout_thirty_board_params = deepcopy(args['board_params'])
    six_mirrored_timeout_thirty_board_params['seed'] = seed
    six_mirrored_timeout_thirty_board_params['n_symbols_min'] = 6
    six_mirrored_timeout_thirty_board_params['n_symbols_max'] = 6
    six_mirrored_timeout_thirty_board_params['timeout'] = 30
    six_mirrored_timeout_thirty_env = RuleBoards(six_mirrored_timeout_thirty_board_params)


    # Full suite
    agent_names = ['candidate', 'oracle', 'random']
    agents = [agent, oracle, random_agent]

    # Needs rerun, should be fast
    for name, agent in zip(agent_names, agents):
        print(f'Working on agent {name}')
        test_suite(agent, only_four_env, OUT_DIR + 'results/' + f'{name}__cond__only_four/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        test_suite(agent, six_mirrored_timeout_thirty_env, OUT_DIR + 'results/' + f'{name}__cond__mirrored_6_timeout_30/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        test_suite(agent, mirrored_five_to_six_env, OUT_DIR + 'results/' + f'{name}__cond__mirrored_five_to_six/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        test_suite(agent, only_three_env, OUT_DIR + 'results/' + f'{name}__cond__only_three/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)

if __name__ == '__main__':
    from multiprocessing import Pool
    from functools import partial
    n = 4
    n_threads = n

    # pool = Pool(n_threads)
    # pool.map(run_one_seed_training, range(n))


    pool = Pool(n_threads)
    pool.map(partial(run_one_seed_testing, n_steps_plot=40, n_steps_total=5000), range(n))

