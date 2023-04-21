"""
See how we can use our saccade + correction setup to do transfer / generalization.

Two envs: one with 1-4 lines, one with only 4 lines 

Two agents, same init.

One is trained only on 4 lines, the other is trained on 1-4 lines.

At the end, compare 4 possibilities:
- trained 1-4 test 1-4 (baseline for success)
- trained 4 test 1-4 (baseline for failure)
- put the 4 lines fovea with the 1-4 periphery (should succeed)
- put the 1-4 lines fovea with the 4 periphery (should fail)

"""
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
from env import Boards
from itertools import product
from oracle import Oracle, RandomAgent
import os
import sys
from tqdm import tqdm

# Technical bother with local vs cluster paths, best solution i could find...
# wdir = '/scratch/atf6569/saccade_drawing/'
# wdir = '/home/arnaud/Scratch/saccade_drawing/'
# wdir = '/scratch/atf6569/saccade_drawing_out/'
# wdir = '/home/arnaud/Scratch/saccade_drawing_out/'
# wdir = '/scratch/atf6569/saccade_drawing_out/'




# OUT_DIR = wdir + 'generalization_n_lines/'   
# OUT_DIR = wdir + 'generalization_n_lines_no_ambiguous_orderings/'
# OUT_DIR = wdir + 'n_lines_generalization/'

# Hopefully, this is the last round of reruns
wdir = '/scratch/atf6569/saccade_drawing/'
OUT_DIR = wdir + 'generalization_n_lines/'   


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

# When removing the fovea, we lose a lot of units, so for fairness increase the size of the peripheral net !
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

    local_ablated_four_or_less_args = deepcopy(one_to_four_args)
    local_ablated_four_or_less_args['seed'] = seed
    local_ablated_four_or_less_args['run_name'] = 'ablated_four_or_less'
    local_ablated_four_or_less_args['agent_params']['fovea_ablated'] = True

    local_ablated_four_or_less_big_peripheral_args = deepcopy(one_to_four_args)
    local_ablated_four_or_less_big_peripheral_args['seed'] = seed
    local_ablated_four_or_less_big_peripheral_args['agent_params']['fovea_ablated'] = True
    local_ablated_four_or_less_big_peripheral_args['agent_params']['peripheral_net_params'] = deepcopy(big_peripheral_net_params)
    local_ablated_four_or_less_big_peripheral_args['run_name'] = 'ablated_four_or_less_big_peripheral'

    local_ablated_four_only_big_peripheral_args = deepcopy(four_lines_args)
    local_ablated_four_only_big_peripheral_args['seed'] = seed
    local_ablated_four_only_big_peripheral_args['agent_params']['fovea_ablated'] = True
    local_ablated_four_only_big_peripheral_args['agent_params']['peripheral_net_params'] = deepcopy(big_peripheral_net_params)
    local_ablated_four_only_big_peripheral_args['run_name'] = 'ablated_four_only_big_peripheral'


    local_four_lines_no_saccades_args = deepcopy(four_lines_args)
    local_four_lines_no_saccades_args['seed'] = seed
    local_four_lines_no_saccades_args['run_name'] = 'no_saccades_four_only'
    local_four_lines_no_saccades_args['agent_params']['no_saccade_loss'] = True

    local_four_or_less_no_saccades_args = deepcopy(one_to_four_args)
    local_four_or_less_no_saccades_args['seed'] = seed
    local_four_or_less_no_saccades_args['run_name'] = 'no_saccades_four_or_less'
    local_four_or_less_no_saccades_args['agent_params']['no_saccade_loss'] = True

    ##################################################################################
    ##################################################################################
    # Finished running on 4/21/23
    # train(local_four_lines_no_saccades_args)
    # train(local_four_or_less_no_saccades_args) 

    
    # train(local_four_lines_ablated_args) # This one was really trash in initial test, low priority  
    # train(local_ablated_four_or_less_args) # Mostly relevant if above works...

    # train(local_four_lines_args)
    # train(local_one_to_four_args)

    # train(local_ablated_four_only_big_peripheral_args)
    # train(local_ablated_four_or_less_big_peripheral_args) 





def test_suite(agent, envs, savepath, oracle, n_steps_plot=100, n_steps_total=1000):
    os.makedirs(savepath, exist_ok=True)

    # Oracle-like means it does not work on observations, but directly on environments
    using_oracle_like = (isinstance(agent, Oracle) or isinstance(agent, RandomAgent))
    # print("Using oracle-like agent:", using_oracle_like)
    # sys.stdout.flush()

    # Obviously plot trajectories
    # Quite a lot of them, but we call only once so it should be fine                   
    test_obs = envs.reset().transpose(0, 3, 1, 2)
    test_done = np.zeros(envs.n_envs)

    print(envs.n_symbols_min, envs.n_symbols_max)

    errors_homing = []
    errors_saccade = []

    # Those will help for nicer visualizations
    magnitude_saccades = []
    magnitude_homings = []
    times = []
    n_lines = []

    # Add some behavioral metrics (namely, tpr and tnr at timeout, since it holds all relevant information)
    tprs = []
    tnrs = []
    reciprocal_overlaps = []

    # On top of the full trajectories, also plot (at most) n_steps_plot final_observations
    n_final_obs_plotted = 0


    for step in tqdm(range(0, n_steps_total)):
        times_step = envs.times.copy()
        # print(times_step.min(), times_step.max())
        n_lines_step = envs.n_symbols.copy()

        oracle_actions, oracle_submoves = oracle.get_action_and_submoves(envs)
        oracle_saccades = oracle_submoves['saccades']

        if not using_oracle_like:
            # Normal networks work on observations
            test_saccades = agent.get_saccade(tch.from_numpy(test_obs).float().to(agent.device)).detach().cpu().numpy()
            test_pos_after_saccade = (envs.positions + test_saccades[:, :2]).copy()
            test_pos_after_saccade = np.clip(test_pos_after_saccade, -1, 1)
            # Just in case, and also to make it explicit we call get_center_patch with float position
            test_fovea_image = envs.get_centered_patches(center_pos=test_pos_after_saccade)
            test_fovea_image = tch.Tensor(np.array([im.transpose(2, 0, 1) for im in test_fovea_image])).float().to(agent.device)
            test_homings = agent.get_homing(test_fovea_image).detach().cpu().numpy()
            test_action = test_saccades + test_homings
        else:
            # Oracle-like agents work on environments
            test_action, test_submoves = agent.get_action_and_submoves(envs)
            test_saccades = test_submoves['saccades']
            if test_saccades.shape[-1] == 2:
                # This happens with oracle 
                test_saccades = np.concatenate([test_saccades, np.zeros((test_saccades.shape[0], 1))], axis=-1)
            test_homings = test_action - test_saccades
            test_pos_after_saccade = (envs.positions + test_saccades[:, :2]).copy()
            test_fovea_image = envs.get_centered_patches(center_pos=test_pos_after_saccade)
            test_fovea_image = tch.Tensor(np.array([im.transpose(2, 0, 1) for im in test_fovea_image])).float()

        # Plots that should happen before the step
        if step < n_steps_plot:

            # TODO: add something to keep a few final obs and their overlaps
            for env_idx in range(2):
                fig, axes = plt.subplots(1, 2, figsize=(32, 16))
                axes[0].imshow(test_obs[env_idx].transpose(2, 1, 0), origin='lower', extent=[-1, 1, -1, 1])
                axes[0].plot([envs.positions[env_idx, 0], envs.positions[env_idx, 0] + test_action[env_idx, 0]], [envs.positions[env_idx, 1], envs.positions[env_idx, 1] + test_action[env_idx, 1]], lw=8, color='gray', label='Total action')
                axes[0].scatter(test_pos_after_saccade[env_idx, 0], test_pos_after_saccade[env_idx, 1], color='m', marker='+', label='Initial saccade')
                axes[0].legend()
                axes[0].set_title('Global image')

                # Plot the fovea image and homing submove 
                axes[1].imshow(test_fovea_image[env_idx].cpu().numpy().transpose(2, 1, 0), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
                axes[1].scatter([test_homings[env_idx, 0]], [test_homings[env_idx, 1]], s=4096, color='m', marker='+', label='Homing submove')
                axes[1].scatter(0, 0, color='r', marker='.', s=4096, label='Eye pos after saccade')
                axes[1].legend()
                axes[1].set_title('Fovea image')

                fig.savefig(savepath + f'traj_{env_idx}_step_{step}.png')
                plt.close(fig)

        # The step itself  
        test_obs, _, test_done, _, test_info = envs.step(test_action, test_done)
        test_obs = test_obs.transpose(0, 3, 1, 2)

        for env_idx in range(envs.n_envs):
            if test_done[env_idx]:
                tprs.append(test_info[env_idx]['tpr'])
                tnrs.append(test_info[env_idx]['tnr'])
                reciprocal_overlaps.append(test_info[env_idx]['reciprocal_overlap'])
                if n_final_obs_plotted < n_steps_plot:
                    n_final_obs_plotted += 1
                    ovlp = test_info[env_idx]['reciprocal_overlap']
                    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
                    ax.imshow(test_info[env_idx]['terminal_observation'].transpose(1, 0, 2), origin='lower', extent=[-1, 1, -1, 1])
                    ax.set_title(f"Final observation, reciprocal_overlaps {ovlp:.3f}")
                    fig.savefig(savepath + f'final_observation_{n_final_obs_plotted}_r_ovlp_{100*round(ovlp, 2)}.png')
                    plt.close(fig)

        # Plots that should happen after the step, only if done
        if step < n_steps_plot:
            for env_idx in range(2):
                if test_done[env_idx]:
                    fig, axes = plt.subplots(1, 2, figsize=(32, 16))
                    axes[0].imshow(test_info[env_idx]['terminal_observation'].transpose(1, 0, 2), origin='lower', extent=[-1, 1, -1, 1])
                    axes[0].set_title('Final observation')

                    # Plot the fovea image and homing submove 
                    axes[1].imshow(np.zeros((32, 32, 3)), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
                    axes[1].set_title('Placeholder fovea image')

                    fig.savefig(savepath + f'traj_{env_idx}_step_{step}_final_observation.png')
                    plt.close(fig)


       
        # errors_homing.append(tch.nn.functional.mse_loss(tch.from_numpy(test_action).float(), tch.from_numpy(oracle_actions).float(), reduction='mean').numpy())
        # errors_saccade.append(tch.nn.functional.mse_loss(tch.from_numpy(test_saccades[:, :2]).float(), tch.from_numpy(oracle_saccades).float(), reduction='mean').numpy())
        errors_homing.append(((test_action-oracle_actions)**2).mean(axis=-1))
        errors_saccade.append(((test_saccades[:,:2]-oracle_saccades)**2).mean(axis=-1))

        # logging.critical(test_saccades[:, :2].shape, errors_saccade[-1].shape)
        magnitude_saccades.append((test_saccades[:, :2]**2).mean(axis=-1))
        magnitude_homings.append((test_homings[:, :2]**2).mean(axis=-1))
        times.append(times_step)
        n_lines.append(n_lines_step)

    errors_homing = np.sqrt(np.array(errors_homing).flatten())  
    errors_saccade = np.sqrt(np.array(errors_saccade).flatten())    

    magnitude_homings = np.sqrt(np.array(magnitude_homings).flatten())
    magnitude_saccades = np.sqrt(np.array(magnitude_saccades).flatten())
    times = np.array(times).flatten()
    n_lines = np.array(n_lines).flatten()

    tprs = np.array(tprs)
    tnrs = np.array(tnrs)
    reciprocal_overlaps = np.array(reciprocal_overlaps)

    # Save the errors for aggregation
    np.save(savepath + f'errors_homing.npy', errors_homing)
    np.save(savepath + f'errors_saccade.npy', errors_saccade)
    np.save(savepath + f'magnitude_saccades.npy', magnitude_saccades)
    np.save(savepath + f'magnitude_homings.npy', magnitude_homings)
    np.save(savepath + f'times.npy', times)
    np.save(savepath + f'n_lines.npy', n_lines)
    np.save(savepath + f'tprs.npy', tprs)
    np.save(savepath + f'tnrs.npy', tnrs)
    np.save(savepath + f'reciprocal_overlaps.npy', reciprocal_overlaps)


    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes[0,0].hist(errors_saccade, bins=40)
    axes[0,0].set_title('Saccade errors histogram')
    axes[0,1].hist(errors_homing, bins=40)
    axes[0,1].set_title('Homing errors histogram')
    axes[1,0].hist(np.log10(errors_saccade+1e-5), bins=40)
    axes[1,0].set_title('Saccade log-errors histogram')
    axes[1,1].hist(np.log10(errors_homing+1e-5), bins=40)
    axes[1,1].set_title('Homing log-errors histogram')
    fig.tight_layout()
    fig.savefig(savepath + f"errors_histogram.png")
                

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes[0,0].hist(errors_saccade, bins=40, log=True)
    axes[0,0].set_title('Saccade errors histogram')
    axes[0,1].hist(errors_homing, bins=40, log=True)
    axes[0,1].set_title('Homing errors histogram')
    axes[1,0].hist(np.log10(errors_saccade+1e-5), bins=40, log=True)
    axes[1,0].set_title('Saccade log-errors histogram')
    axes[1,1].hist(np.log10(errors_homing+1e-5), bins=40, log=True)
    axes[1,1].set_title('Homing log-errors histogram')
    fig.tight_layout()
    fig.savefig(savepath + f"errors_log_histogram.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # How much os the lines is covered by the drawing (truly between 0 and 1)
    axes[0].hist(tprs, bins=50, range=[0,1], label=f'Min {tprs.min():.2f}; Max {tprs.max():.2f}, std {tprs.std():.2f}')
    axes[0].set_xlim([-0.05, 1.05])
    axes[0].legend()
    axes[0].set_title('Final TPR histogram')

    # How much coloring was done outside the lines (a priori between 0 and 1, but very close to 0 so allow scale change)
    axes[1].hist(1.-tnrs, bins=50, label=f'Min {(1.-tnrs).min():.2f}; Max {(1.- tnrs).max():.2f}, std {tnrs.std():.2f}')
    axes[1].set_title('Final FNR histogram')
    axes[1].legend()

    # This one is also between 0 and 1, but does not really define an "order" (because of how it's calculated, there could be local optima)
    # In practice, along with the other two metrics, it's a good indicator of how well the model is doing
    axes[2].hist(reciprocal_overlaps, bins=50, range=[0,1], label=f'Min {reciprocal_overlaps.min():.2f}; Max {reciprocal_overlaps.max():.2f}, std {reciprocal_overlaps.std():.2f}')
    axes[2].set_xlim([-0.05, 1.05])
    axes[2].set_title('Reciprocal overlap histogram')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(savepath + f"tpr_tnr_histogram.png")

    
    df = pd.DataFrame({'times': times.astype(int), 'errors_homing': errors_homing, 'log_errors_homing': np.log10(errors_homing+1e-5), 'errors_saccade': errors_saccade, 'log_errors_saccade': np.log10(errors_saccade+1e-5),
                        'magnitude_saccades': magnitude_saccades, 'log_magnitude_saccades': np.log10(magnitude_saccades+1e-5), 'magnitude_homings': magnitude_homings, 'log_magnitude_homings': np.log10(magnitude_homings+1e-5), 'n_lines': n_lines.astype(int)})
    
    try:
        # This one is for understanding the patterns of errors
        fig, axes = plt.subplots(2, 3, figsize=(30, 10))
        sns.scatterplot(data=df, x='magnitude_saccades', y='errors_saccade', ax=axes[0,0])
        axes[0,0].set_title('Saccade errors vs magnitude')
        axes[0,0].set_rasterized(True)

        sns.scatterplot(data=df, x='magnitude_homings', y='errors_homing', ax=axes[1,0])
        axes[1,0].set_title('Homing errors vs magnitude')
        axes[1,0].set_rasterized(True)

        sns.violinplot(data=df, x='times', y='log_errors_saccade', ax=axes[0,1])
        axes[0, 1].set_title('Saccade log-errors vs time in trajectory')
        axes[0,1].set_rasterized(True)

        sns.violinplot(data=df, x='times', y='log_errors_homing', ax=axes[1,1])
        axes[1,1].set_title('Homing log-errors vs time in trajectory')
        axes[1,1].set_rasterized(True)

        sns.violinplot(data=df, x='n_lines', y='log_errors_saccade', ax=axes[0,2])
        axes[0, 2].set_title('Saccade log-errors vs number of lines')
        axes[0,2].set_rasterized(True)

        sns.violinplot(data=df, x='n_lines', y='log_errors_homing', ax=axes[1,2])
        axes[1,2].set_title('Homing errors vs number of lines')
        axes[1,2].set_rasterized(True)

        fig.tight_layout()
        fig.savefig(savepath + f"patterns_of_errors.pdf")
    except:
        raise RuntimeError('Error plotting patterns of errors')

    # No memory leaks on my watch !    
    plt.close('all')


def run_one_seed_testing(seed, n_steps_plot=10, n_steps_total=100):
    # Do the network frankensteining / testing
    four_lines_agent = SaccadeAgent(four_lines_args['agent_params']['peripheral_net_params'], four_lines_args['agent_params']['foveal_net_params'])
    four_lines_agent.load_state_dict(tch.load(OUT_DIR + f'only_four/seed{seed}/final_agent.pt'))

    one_to_four_agent = SaccadeAgent(one_to_four_args['agent_params']['peripheral_net_params'], one_to_four_args['agent_params']['foveal_net_params'])
    one_to_four_agent.load_state_dict(tch.load(OUT_DIR + f'four_or_less/seed{seed}/final_agent.pt'))

    local_ablated_four_or_less_args = deepcopy(one_to_four_args)
    local_ablated_four_or_less_args['seed'] = seed
    local_ablated_four_or_less_args['run_name'] = 'ablated_four_or_less'
    local_ablated_four_or_less_args['agent_params']['fovea_ablated'] = True
    ablated_four_agent = SaccadeAgent(four_lines_args['agent_params']['peripheral_net_params'], four_lines_args['agent_params']['foveal_net_params'])
    ablated_four_agent.load_state_dict(tch.load(OUT_DIR + f'ablated_four_only/seed{seed}/final_agent.pt'))

    ablated_four_or_less_agent = SaccadeAgent(one_to_four_args['agent_params']['peripheral_net_params'], one_to_four_args['agent_params']['foveal_net_params'])
    ablated_four_or_less_agent.load_state_dict(tch.load(OUT_DIR + f'ablated_four_or_less/seed{seed}/final_agent.pt'))

    # Still running due to human error
    ablated_four_or_less_big_peripheral_agent = SaccadeAgent(deepcopy(big_peripheral_net_params), one_to_four_args['agent_params']['foveal_net_params'])
    ablated_four_or_less_big_peripheral_agent.load_state_dict(tch.load(OUT_DIR + f'ablated_four_or_less_big_peripheral/seed{seed}/final_agent.pt'))


    ablated_four_only_big_peripheral_args = deepcopy(four_lines_args)
    ablated_four_only_big_peripheral_args['seed'] = seed
    ablated_four_only_big_peripheral_args['agent_params']['fovea_ablated'] = True
    ablated_four_only_big_peripheral_args['agent_params']['peripheral_net_params'] = deepcopy(big_peripheral_net_params)
    ablated_four_only_big_peripheral_args['run_name'] = 'ablated_four_only_big_peripheral'

    four_lines_no_saccades_args = deepcopy(four_lines_args)
    four_lines_no_saccades_args['seed'] = seed
    four_lines_no_saccades_args['run_name'] = 'no_saccades_four_only'
    four_lines_no_saccades_args['agent_params']['no_saccade_loss'] = True

    four_or_less_no_saccades_args = deepcopy(one_to_four_args)
    four_or_less_no_saccades_args['seed'] = seed
    four_or_less_no_saccades_args['run_name'] = 'no_saccades_four_or_less'
    four_or_less_no_saccades_args['agent_params']['no_saccade_loss'] = True

    ablated_four_only_big_peripheral_agent = SaccadeAgent(deepcopy(big_peripheral_net_params), four_lines_args['agent_params']['foveal_net_params'])
    ablated_four_only_big_peripheral_agent.load_state_dict(tch.load(OUT_DIR + f'ablated_four_only_big_peripheral/seed{seed}/final_agent.pt'))

    four_lines_no_saccades_agent = SaccadeAgent(four_lines_args['agent_params']['peripheral_net_params'], four_lines_args['agent_params']['foveal_net_params'])
    four_lines_no_saccades_agent.load_state_dict(tch.load(OUT_DIR + f'no_saccades_four_only/seed{seed}/final_agent.pt'))

    four_or_less_no_saccades_agent = SaccadeAgent(one_to_four_args['agent_params']['peripheral_net_params'], one_to_four_args['agent_params']['foveal_net_params'])
    four_or_less_no_saccades_agent.load_state_dict(tch.load(OUT_DIR + f'no_saccades_four_or_less/seed{seed}/final_agent.pt'))

    oracle = Oracle(**four_lines_args['oracle_params']) # Oracle does not care for number of lines, it reads it from the environment
    random_agent = RandomAgent(amplitude=.5, seed=seed)

    # # This is just a sanity check to ensure we are not using the same network for frankensteining, which would not be very interesting
    # assert not tch.allclose(four_lines_agent.peripheral_net.convnet[0].weight, one_to_four_agent.peripheral_net.convnet[0].weight)

    ref_board_params = deepcopy(four_lines_args['board_params'])

    only_two_board_params = deepcopy(four_lines_args['board_params'])
    only_two_board_params['seed'] = seed
    only_two_board_params['n_symbols_min'] = 2
    only_two_board_params['n_symbols_max'] = 2
    only_two_env = Boards(only_two_board_params)

    only_three_board_params = deepcopy(four_lines_args['board_params'])
    only_three_board_params['seed'] = seed
    only_three_board_params['n_symbols_min'] = 3
    only_three_board_params['n_symbols_max'] = 3
    only_three_env = Boards(only_three_board_params)

    only_four_env = Boards(four_lines_args['board_params'])

    mirrored_one_to_four_args = deepcopy(one_to_four_args)
    mirrored_one_to_four_args['seed'] = seed

    one_to_four_env = Boards(one_to_four_args['board_params'])
    five_to_six_env = Boards(five_to_six_args['board_params'])
    mirrored_five_to_six_env = Boards(mirrored_five_to_six_args['board_params'])
    


    # agent_names = ['four_lines_no_saccades', 'four_or_less_no_saccades', 'four_only', 'ablated_four_only', 'ablated_four_or_less', 'ablated_four_only_big_peripheral', 'four_or_less', 'ablated_four_or_less_big_peripheral']
    # agents = [four_lines_no_saccades_agent, four_or_less_no_saccades_agent, four_lines_agent, ablated_four_agent, ablated_four_or_less_agent, ablated_four_only_big_peripheral_agent, one_to_four_agent, ablated_four_or_less_big_peripheral_agent]

    # agent_names = ['four_or_less']
    # agents = [one_to_four_agent]

    # Still todo, important as reference
    # agent_names = ['oracle']
    # agents = [oracle]

    agent_names = ['ablated_four_only_big_peripheral']
    agents = [ablated_four_only_big_peripheral_agent]

    for name, agent in zip(agent_names, agents):
    # for name, agent in zip(['four_or_less', 'only_four'], [one_to_four_agent, four_lines_agent]):
        print(f'Working on agent {name}')
        # test_suite(agent, only_four_env, OUT_DIR + 'results/' + f'{name}__cond__only_four/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        # test_suite(agent, one_to_four_env, OUT_DIR + 'results/' + f'{name}__cond__four_or_less/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        # test_suite(agent, five_to_six_env, OUT_DIR + 'results/' + f'{name}__cond__five_to_six/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        test_suite(agent, mirrored_five_to_six_env, OUT_DIR + 'results/' + f'{name}__cond__mirrored_five_to_six/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)

        # test_suite(agent, only_three_env, OUT_DIR + 'results/' + f'{name}__cond__only_three/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)
        # test_suite(agent, only_two_env, OUT_DIR + 'results/' + f'{name}__cond__only_two/{seed}/', oracle, n_steps_plot=n_steps_plot, n_steps_total=n_steps_total)


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