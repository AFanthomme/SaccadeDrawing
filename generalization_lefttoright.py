# Should rerun after closest, it comes from older versions of the code !

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

# # Technical bother with local vs cluster paths...
wdir = '/scratch/atf6569/saccade_drawing/'
OUT_DIR = wdir + 'generalization_lefttoright/'   


one_to_four_args = {
        'root_output_folder': OUT_DIR,
        'run_name': 'four_or_less',

        'board_params': {
            'n_envs': 64,
            'n_symbols_min': 1,
            'n_symbols_max': 4,
            'reward_type': 'default',
            'reward_params':  {'overlap_criterion': .2},
            'ordering_sensitivity': 0, # in pixels
            'all_envs_start_identical': False, 
            'timeout': None,
            'mirror': False,
            'allowed_symbols': ['line_0', 'line_1', 'line_2', 'line_3'],
            'allowed_rules': ['rightward',],
        },

        'oracle_params': {
            'sensitivity': .1,
            'noise': .02,
        },

        'agent_params': {
            'fovea_ablated': False,
            'no_saccade_loss': False,
            'peripheral_net_params': {
                'n_pixels_in': 128,
                'cnn_n_featuremaps': [64, 64, 64, 32, 32],
                'cnn_kernel_sizes': [5, 5, 3, 3, 3],
                'cnn_kernel_strides': [1, 1, 1, 1, 1],
                'fc_sizes': [256, 256],
                'rnn_size': 256,
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
            # 'n_updates': 5_000,
            'n_updates': 10_000,
            # 'test_every': 500,
            'test_every': 1000,
            'lr': 5e-4,
            # 'seed': seed,
            # 'batch_size': 64,
            'batch_size': 128,
            # How many env steps we do collecting data before updating the agent
            # 'rollout_length': 10,
            'rollout_length': 20,
            # How many batches / GD steps we do after each rollout phase 
            'gd_steps_per_rollout': 5, # about rollout steps * n_envs / batch_size
            # 'start_agent_only_after': 1_000,
            # 'start_agent_only_after': 2_000,
            'start_agent_only_after': 500, # This is kind of a weird crutch, try to get rid of it !
        },
    }

# # When removing the fovea, we lose a lot of units, so for fairness increase the size of the peripheral net !
big_peripheral_net_params = {
    'n_pixels_in': 128,
    'cnn_n_featuremaps': [128, 128, 128, 64, 64],
    'cnn_kernel_sizes': [5, 5, 3, 3, 3],
    'cnn_kernel_strides': [1, 1, 1, 1, 1],
    'fc_sizes': [512, 512],
    'rnn_size': 512,
    'recurrence_type': 'rnn',
    'recurrent_steps': 1, 
    }


four_lines_args = deepcopy(one_to_four_args)
four_lines_args['board_params']['n_symbols_min'] = 4
four_lines_args['board_params']['n_symbols_max'] = 4
four_lines_args['run_name'] = 'only_four'

# Those are not used for training, hence don't need run names but put them just in case to avoid overwriting
five_to_six_args = deepcopy(four_lines_args)
five_to_six_args['board_params']['n_symbols_min'] = 5
five_to_six_args['board_params']['n_symbols_max'] = 6
five_to_six_args['board_params']['timeout'] = 15 # Just in case we can recover; helps, but we fail so rarely it's not very relevant 
five_to_six_args['run_name'] = 'five_to_six'

mirrored_five_to_six_args = deepcopy(five_to_six_args)
mirrored_five_to_six_args['board_params']['mirror'] = True
mirrored_five_to_six_args['run_name'] = 'mirrored_one_to_six'

def run_one_seed_training(seed):
    local_one_to_four_args = deepcopy(one_to_four_args)
    local_one_to_four_args['seed'] = seed

    local_four_lines_args = deepcopy(four_lines_args)
    local_four_lines_args['seed'] = seed

    # Put some more effort on ablated nets as they are our "stat-of-the-art" 
    # and we want to make sure we give them a fighting chance
    local_four_lines_ablated_args = deepcopy(four_lines_args)
    local_four_lines_ablated_args['seed'] = seed
    local_four_lines_ablated_args['run_name'] = 'ablated_four_only'
    local_four_lines_ablated_args['agent_params']['fovea_ablated'] = True

    local_ablated_four_only_big_peripheral_args = deepcopy(four_lines_args)
    local_ablated_four_only_big_peripheral_args['seed'] = seed
    local_ablated_four_only_big_peripheral_args['agent_params']['fovea_ablated'] = True
    local_ablated_four_only_big_peripheral_args['agent_params']['peripheral_net_params'] = deepcopy(big_peripheral_net_params)
    local_ablated_four_only_big_peripheral_args['run_name'] = 'ablated_four_only_big_peripheral'


    local_four_lines_no_saccades_args = deepcopy(four_lines_args)
    local_four_lines_no_saccades_args['seed'] = seed
    local_four_lines_no_saccades_args['run_name'] = 'no_saccades_four_only'
    local_four_lines_no_saccades_args['agent_params']['no_saccade_loss'] = True


    ##################################################################################
    ##################################################################################
    # train(local_four_lines_no_saccades_args)
    
    # train(local_four_lines_ablated_args) # This one was really trash in initial test, low priority  

    # train(local_four_lines_args)

    train(local_ablated_four_only_big_peripheral_args)





def run_one_seed_testing(seed, n_steps_plot=10, n_steps_total=100):
    # Still running !
    four_lines_agent = SaccadeAgent(four_lines_args['agent_params']['peripheral_net_params'], four_lines_args['agent_params']['foveal_net_params'])
    four_lines_agent.load_state_dict(tch.load(OUT_DIR + f'only_four/seed{seed}/final_agent.pt'))

    ablated_four_agent = SaccadeAgent(four_lines_args['agent_params']['peripheral_net_params'], four_lines_args['agent_params']['foveal_net_params'])
    ablated_four_agent.load_state_dict(tch.load(OUT_DIR + f'ablated_four_only/seed{seed}/final_agent.pt'))

    # Still running !
    ablated_four_only_big_peripheral_args = deepcopy(four_lines_args)
    ablated_four_only_big_peripheral_args['seed'] = seed
    ablated_four_only_big_peripheral_args['agent_params']['fovea_ablated'] = True
    ablated_four_only_big_peripheral_args['agent_params']['peripheral_net_params'] = deepcopy(big_peripheral_net_params)
    ablated_four_only_big_peripheral_args['run_name'] = 'ablated_four_only_big_peripheral'
    ablated_four_only_big_peripheral_agent = SaccadeAgent(deepcopy(big_peripheral_net_params), four_lines_args['agent_params']['foveal_net_params'])
    ablated_four_only_big_peripheral_agent.load_state_dict(tch.load(OUT_DIR + f'ablated_four_only_big_peripheral/seed{seed}/final_agent.pt'))

    four_lines_no_saccades_args = deepcopy(four_lines_args)
    four_lines_no_saccades_args['seed'] = seed
    four_lines_no_saccades_args['run_name'] = 'no_saccades_four_only'
    four_lines_no_saccades_args['agent_params']['no_saccade_loss'] = True
    four_lines_no_saccades_agent = SaccadeAgent(four_lines_args['agent_params']['peripheral_net_params'], four_lines_args['agent_params']['foveal_net_params'])
    four_lines_no_saccades_agent.load_state_dict(tch.load(OUT_DIR + f'no_saccades_four_only/seed{seed}/final_agent.pt'))

    oracle = RuleOracle(**four_lines_args['oracle_params']) # RuleOracle does not care for number of lines, it reads it from the environment
    random_agent = RandomAgent(amplitude=.5, seed=seed)

    # # This is just a sanity check to ensure we are not using the same network for frankensteining, which would not be very interesting
    # assert not tch.allclose(four_lines_agent.peripheral_net.convnet[0].weight, one_to_four_agent.peripheral_net.convnet[0].weight)

    ref_board_params = deepcopy(four_lines_args['board_params'])

    only_two_board_params = deepcopy(four_lines_args['board_params'])
    only_two_board_params['seed'] = seed
    only_two_board_params['n_symbols_min'] = 2
    only_two_board_params['n_symbols_max'] = 2
    only_two_env = RuleBoards(only_two_board_params)

    only_three_board_params = deepcopy(four_lines_args['board_params'])
    only_three_board_params['seed'] = seed
    only_three_board_params['n_symbols_min'] = 3
    only_three_board_params['n_symbols_max'] = 3
    only_three_env = RuleBoards(only_three_board_params)

    only_four_env = RuleBoards(four_lines_args['board_params'])

    mirrored_one_to_four_args = deepcopy(one_to_four_args)
    mirrored_one_to_four_args['seed'] = seed

    five_to_six_env = RuleBoards(five_to_six_args['board_params'])
    mirrored_five_to_six_env = RuleBoards(mirrored_five_to_six_args['board_params'])
    
    # This one will let us see if giving it more time helps it get over such a big generalization leap
    eight_timeout_thirty_board_params = deepcopy(five_to_six_args['board_params'])
    eight_timeout_thirty_board_params['seed'] = seed
    eight_timeout_thirty_board_params['n_symbols_min'] = 8
    eight_timeout_thirty_board_params['n_symbols_max'] = 8
    eight_timeout_thirty_board_params['timeout'] = 30
    eight_timeout_thirty_env = RuleBoards(eight_timeout_thirty_board_params)

    six_mirrored_timeout_thirty_board_params = deepcopy(five_to_six_args['board_params'])
    six_mirrored_timeout_thirty_board_params['seed'] = seed
    six_mirrored_timeout_thirty_board_params['n_symbols_min'] = 6
    six_mirrored_timeout_thirty_board_params['n_symbols_max'] = 6
    six_mirrored_timeout_thirty_board_params['timeout'] = 30
    six_mirrored_timeout_thirty_env = RuleBoards(six_mirrored_timeout_thirty_board_params)

    # Full suite
    # agent_names = ['oracle', 'random', 'candidate', 'ablated', 'big_ablated', 'saccade_constrained']
    # agents = [oracle, random_agent, four_lines_no_saccades_agent, ablated_four_agent, ablated_four_only_big_peripheral_agent]

    # First wave of runs
    agent_names = ['candidate', 'ablated', 'oracle', 'random']#, 'big_ablated', 'saccade_constrained']
    agents = [four_lines_no_saccades_agent, ablated_four_agent, oracle, random_agent]#, ablated_four_only_big_peripheral_agent, four_lines_agent]

    # Second wave, not launched yet
    # agent_names = ['big_ablated', 'saccade_constrained']
    # agents = [ablated_four_only_big_peripheral_agent, four_lines_agent,]


    # Quick rerun,6_30 did not run for saccade_constrained

    agent_names = ['saccade_constrained']
    agents = [four_lines_agent]

    for name, agent in zip(agent_names, agents):
        print(f'Working on agent {name}')
        # test_suite(agent, only_four_env, OUT_DIR + 'results/' + f'{name}__cond__only_four/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        # test_suite(agent, five_to_six_env, OUT_DIR + 'results/' + f'{name}__cond__five_to_six/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        # test_suite(agent, mirrored_five_to_six_env, OUT_DIR + 'results/' + f'{name}__cond__mirrored_five_to_six/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        # test_suite(agent, only_three_env, OUT_DIR + 'results/' + f'{name}__cond__only_three/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        # test_suite(agent, only_two_env, OUT_DIR + 'results/' + f'{name}__cond__only_two/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        # test_suite(agent, eight_timeout_thirty_env, OUT_DIR + 'results/' + f'{name}__cond__eight_timeout_thirty/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        test_suite(agent, six_mirrored_timeout_thirty_env, OUT_DIR + 'results/' + f'{name}__cond__six_mirrored_timeout_thirty/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)



if __name__ == '__main__':
    from multiprocessing import Pool
    from functools import partial
    n = 4
    n_threads = n
    # n_threads = 1

    # pool = Pool(n_threads)
    # pool.map(run_one_seed_training, range(n))

#     # for i in range(n):
#     #     run_one_seed_testing(i, n_steps_plot=40, n_steps_total=5000)

    pool = Pool(n_threads)
    pool.map(partial(run_one_seed_testing, n_steps_plot=40, n_steps_total=5000), range(n))

    # Quick run for prototyping
    # pool.map(partial(run_one_seed_testing, n_steps_plot=40, n_steps_total=500), range(n))