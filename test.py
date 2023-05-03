'''
Test suite common to all the modules, mostly does basic stuff
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

def test_suite(agent, envs, savepath, oracle, n_steps_plot=100, n_steps_total=1000):
    os.makedirs(savepath, exist_ok=True)

    # RuleOracle-like means it does not work on observations, but directly on environments
    using_oracle_like = (isinstance(agent, RuleOracle) or isinstance(agent, RandomAgent))
    # print("Using oracle-like agent:", using_oracle_like)
    # sys.stdout.flush()

    # Obviously plot trajectories
    # Quite a lot of them, but we call only once so it should be fine                   
    test_obs = envs.reset().transpose(0, 3, 1, 2)
    test_done = np.zeros(envs.n_envs)

    print(envs.n_symbols_min, envs.n_symbols_max)

    errors_homing = []
    errors_saccade = []

    # Embarassingly forgot to save those...
    all_saccades = []
    all_homings = []
    all_actions = []
    all_start_positions = []
    all_oracle_saccades = []
    all_oracle_actions = []

    # Those will help for nicer visualizations
    magnitude_saccades = []
    magnitude_homings = []
    times = []
    n_lines = []

    # Those are new with the rulebased envs
    rules = []
    rewards = []
    cumulated_rewards = [] # Will require some caution, but avoid stitching shenanigans in aggregation; worst case, use rewards themselves
    running_overlap = []

    # Add some behavioral metrics (namely, tpr and tnr at timeout, since it holds all relevant information)
    tprs = []
    tnrs = []
    reciprocal_overlaps = []

    # On top of the full trajectories, also plot (at most) n_steps_plot final_observations
    n_final_obs_plotted = 0

    # Before starting, no cumulated rewards
    cumulated_rewards_step = np.zeros(envs.n_envs)
    for step in tqdm(range(0, n_steps_total)):
        times_step = envs.times.copy()
        # print(times_step.min(), times_step.max())
        n_lines_step = envs.n_symbols.copy()
        rules_step = envs.rules.copy() # object array, rule stored as string
        start_positions_step = envs.positions.copy()

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
            # RuleOracle-like agents work on environments
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
        test_obs, test_rewards, test_done, _, test_info = envs.step(test_action, test_done)
        test_obs = test_obs.transpose(0, 3, 1, 2)


        running_overlap_step = [info['reciprocal_overlap'] for info in test_info]

        cumulated_rewards_step += test_rewards
        next_cumulated_rewards_step = cumulated_rewards_step.copy()

        for env_idx in range(envs.n_envs):
            if test_done[env_idx]:
                tprs.append(test_info[env_idx]['tpr'])
                tnrs.append(test_info[env_idx]['tnr'])
                reciprocal_overlaps.append(test_info[env_idx]['reciprocal_overlap'])
                next_cumulated_rewards_step[env_idx] = 0

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

        # A priori, all of those will be "aligned" (same siz, and a given index corresponds to the same "step" for everyone)
        magnitude_saccades.append((test_saccades[:, :2]**2).mean(axis=-1))
        magnitude_homings.append((test_homings[:, :2]**2).mean(axis=-1))
        times.append(times_step)
        n_lines.append(n_lines_step)
        rewards.append(test_rewards)
        rules.append(rules_step)
        cumulated_rewards.append(cumulated_rewards_step)
        running_overlap.append(running_overlap_step)
        all_oracle_actions.append(oracle_actions.copy())
        all_oracle_saccades.append(oracle_saccades.copy())
        all_saccades.append(test_saccades.copy())
        all_homings.append(test_homings.copy())
        all_start_positions.append(start_positions_step)
        all_actions.append(test_action.copy())

        # Now that we saved it, put the new one that has reset to 0 in done envs
        cumulated_rewards_step = next_cumulated_rewards_step.copy()

    # errors_homing = np.sqrt(np.array(errors_homing).flatten())  
    # errors_saccade = np.sqrt(np.array(errors_saccade).flatten())    
    # magnitude_homings = np.sqrt(np.array(magnitude_homings).flatten())
    # magnitude_saccades = np.sqrt(np.array(magnitude_saccades).flatten())
    # times = np.array(times).flatten()
    # n_lines = np.array(n_lines).flatten()
    # rewards = np.array(rewards).flatten()
    # rules = np.array(rules).flatten()
    # cumulated_rewards = np.array(cumulated_rewards).flatten()
    # running_overlap = np.array(running_overlap).flatten()
    # all_oracle_actions = np.array(all_oracle_actions).reshape(-1, 2)
    # all_oracle_saccades = np.array(all_oracle_saccades).reshape(-1, 2)
    # all_saccades = np.array(all_saccades).reshape(-1, 3)
    # all_homings = np.array(all_homings).reshape(-1, 3)
    # all_actions = np.array(all_actions).reshape(-1, 3)
    # all_start_positions = np.array(all_start_positions).reshape(-1, 2)

    errors_homing = np.sqrt(np.array(errors_homing))  
    errors_saccade = np.sqrt(np.array(errors_saccade))    
    magnitude_homings = np.sqrt(np.array(magnitude_homings))
    magnitude_saccades = np.sqrt(np.array(magnitude_saccades))
    times = np.array(times)
    n_lines = np.array(n_lines)
    rewards = np.array(rewards)
    rules = np.array(rules)
    cumulated_rewards = np.array(cumulated_rewards)
    running_overlap = np.array(running_overlap)
    all_oracle_actions = np.array(all_oracle_actions)
    all_oracle_saccades = np.array(all_oracle_saccades)
    all_saccades = np.array(all_saccades)
    all_homings = np.array(all_homings)
    all_actions = np.array(all_actions)
    all_start_positions = np.array(all_start_positions)

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
    np.save(savepath + f'rewards.npy', rewards)
    np.save(savepath + f'rules.npy', rules)
    np.save(savepath + f'cumulated_rewards.npy', cumulated_rewards)
    np.save(savepath + f'running_overlap.npy', running_overlap)
    np.save(savepath + f'all_oracle_actions.npy', all_oracle_actions)
    np.save(savepath + f'all_oracle_saccades.npy', all_oracle_saccades)
    np.save(savepath + f'all_saccades.npy', all_saccades)
    np.save(savepath + f'all_homings.npy', all_homings)
    np.save(savepath + f'all_actions.npy', all_actions)
    np.save(savepath + f'all_start_positions.npy', all_start_positions)

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes[0,0].hist(errors_saccade.flatten(), bins=40)
    axes[0,0].set_title('Saccade errors histogram')
    axes[0,1].hist(errors_homing.flatten(), bins=40)
    axes[0,1].set_title('Homing errors histogram')
    axes[1,0].hist(np.log10(errors_saccade.flatten()+1e-5), bins=40)
    axes[1,0].set_title('Saccade log-errors histogram')
    axes[1,1].hist(np.log10(errors_homing.flatten()+1e-5), bins=40)
    axes[1,1].set_title('Homing log-errors histogram')
    fig.tight_layout()
    fig.savefig(savepath + f"errors_histogram.png")
                

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes[0,0].hist(errors_saccade.flatten(), bins=40, log=True)
    axes[0,0].set_title('Saccade errors histogram')
    axes[0,1].hist(errors_homing, bins=40, log=True)
    axes[0,1].set_title('Homing errors histogram')
    axes[1,0].hist(np.log10(errors_saccade.flatten()+1e-5), bins=40, log=True)
    axes[1,0].set_title('Saccade log-errors histogram')
    axes[1,1].hist(np.log10(errors_homing.flatten()+1e-5), bins=40, log=True)
    axes[1,1].set_title('Homing log-errors histogram')
    fig.tight_layout()
    fig.savefig(savepath + f"errors_log_histogram.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # How much os the lines is covered by the drawing (truly between 0 and 1)
    axes[0].hist(tprs.flatten(), bins=50, range=[0,1], label=f'Min {tprs.min():.2f}; Max {tprs.max():.2f}, std {tprs.std():.2f}')
    axes[0].set_xlim([-0.05, 1.05])
    axes[0].legend()
    axes[0].set_title('Final TPR histogram')

    # How much coloring was done outside the lines (a priori between 0 and 1, but very close to 0 so allow scale change)
    axes[1].hist(1.-tnrs.flatten(), bins=50, label=f'Min {(1.-tnrs).min():.2f}; Max {(1.- tnrs).max():.2f}, std {tnrs.std():.2f}')
    axes[1].set_title('Final FNR histogram')
    axes[1].legend()

    # This one is also between 0 and 1, but does not really define an "order" (because of how it's calculated, there could be local optima)
    # In practice, along with the other two metrics, it's a good indicator of how well the model is doing
    axes[2].hist(reciprocal_overlaps.flatten(), bins=50, range=[0,1], label=f'Min {reciprocal_overlaps.min():.2f}; Max {reciprocal_overlaps.max():.2f}, std {reciprocal_overlaps.std():.2f}')
    axes[2].set_xlim([-0.05, 1.05])
    axes[2].set_title('Reciprocal overlap histogram')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(savepath + f"tpr_tnr_histogram.png")

    df = pd.DataFrame({'times': times.flatten().astype(int), 'errors_homing': errors_homing.flatten(), 'log_errors_homing': np.log10(errors_homing.flatten()+1e-5), 'errors_saccade': errors_saccade.flatten(), 'log_errors_saccade': np.log10(errors_saccade.flatten()+1e-5),
                        'magnitude_saccades': magnitude_saccades.flatten(), 'log_magnitude_saccades': np.log10(magnitude_saccades.flatten()+1e-5), 'magnitude_homings': magnitude_homings.flatten(), 'log_magnitude_homings': np.log10(magnitude_homings.flatten()+1e-5), 'n_lines': n_lines.flatten().astype(int)})
    
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
