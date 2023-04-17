from buffer import Buffer
from agent import SaccadeAgent
from oracle import Oracle
from env import Boards

import torch as tch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import os
import time
import logging
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from copy import deepcopy


def train(args):
    args = SimpleNamespace(**args)

    run_name = args.run_name + f"/seed{args.seed}"

    # Set up output folder
    root_output_folder = args.root_output_folder
    savepath = root_output_folder + run_name + "/"

    os.makedirs(root_output_folder, exist_ok=True)
    writer = SummaryWriter(root_output_folder + f"tensorboard/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Important: seeding
    seed = args.seed
    np.random.seed(seed)
    tch.manual_seed(seed)
    tch.backends.cudnn.deterministic = True

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Starting training')

    # Set up objects
    envs = Boards(args.board_params)
    tmp = {k: v for k, v in args.board_params.items()}
    tmp['all_envs_start_identical'] = False
    test_envs = Boards(tmp)
    oracle = Oracle(**args.oracle_params)
    agent = SaccadeAgent(args.agent_params['peripheral_net_params'], args.agent_params['foveal_net_params'], fovea_ablated=args.agent_params['fovea_ablated'])
    buffer = Buffer(args.buffer_params['buffer_size'])
    foveal_optimizer = tch.optim.Adam(agent.foveal_net.parameters(), lr=args.training_params['lr'])
    peripheral_optimizer = tch.optim.Adam(agent.peripheral_net.parameters(), lr=args.training_params['lr'])


    # This comes from buffer.py 
    # Keep a few sanity checks here just in case
    obs = envs.reset().transpose(0, 3, 1, 2)

    checked_buffer=False
    for update in range(args.training_params['n_updates']):

        ###########################################################
        #################                         #################
        #################      Rollout phase      #################
        #################                         #################
        ###########################################################

        this_rollout_rng = np.random.rand()
        for rollout_step in range(args.training_params['rollout_length']):
            transition_dict = {}

            # Info about the inital observation
            transition_dict['obs'] = obs.copy()
            transition_dict['endpoints'] = envs.target_endpoints.copy()
            transition_dict['positions'] = envs.positions.copy()
            transition_dict['symbols_done'] = envs.symbols_done.copy()
            transition_dict['n_symbols'] = envs.n_symbols.copy()
            transition_dict['barycenters'] = envs.barycenters.copy()

            # Info about oracle behavior
            oracle_actions, oracle_submoves = oracle.get_action_and_submoves(envs)
            oracle_saccades = oracle_submoves['saccades']
            transition_dict['oracle_actions'] = oracle_actions.copy()
            transition_dict['oracle_saccades'] = oracle_saccades.copy()
            oracle_pos_after_saccade = (envs.positions + oracle_saccades).copy()
            oracle_pos_after_saccade = np.clip(oracle_pos_after_saccade, -1, 1)
            fovea_image = envs.get_centered_patches(center_pos=oracle_pos_after_saccade)
            fovea_image = [im.transpose(2, 0, 1) for im in fovea_image]

            with tch.no_grad():
                # Get the agent saccade; no need to save it here, will be recomputed in grad_enabled mode
                saccades = agent.get_saccade(tch.from_numpy(obs).float().to(agent.device)).detach().cpu().numpy()
                foveas = envs.get_centered_patches(center_pos=envs.positions + saccades[:, :2])
                foveas = tch.Tensor(np.array([im.transpose(2, 0, 1) for im in foveas])).float().to(agent.device)
                homings =  agent.get_homing(foveas).detach().cpu().numpy()

                # Saccade does not allow writing on the page, only homing does
                # If fovea_ablated, it would prevent the agent from writing at all so skip !!!
                if not args.agent_params['fovea_ablated']:
                    saccades[:, -1] = 0. 
                agent_actions = saccades + homings
            
            transition_dict['fovea_after_saccade'] = foveas.detach().cpu().numpy().copy()

            #############################################################################
            #############################################################################
            #############################################################################

            # TODO: Remove all of this if we can, it's a crutch and makes comparison between models a bit muddy
            # First try "start_agent_only_after" very small, if it orks cut that completely

            # Choose how actions are taken in the environment for the entire rollout
            # 4 possibilities:
            # - oracle_actions (50%)
            # - 80% oracle, 20 % agent (25%)
            # - 50% oracle, 50 % agent (15%)
            # - agent + noise (10%)

            if update >= args.training_params['start_agent_only_after']:
                actions = agent_actions.copy()
            else:
                if this_rollout_rng < 0.5:
                    actions = oracle_actions.copy()
                elif this_rollout_rng < 0.75:
                    if np.random.rand() < 0.8:
                        actions = oracle_actions.copy()
                    else:
                        actions = agent_actions.copy()
                elif this_rollout_rng < 0.9:
                    if np.random.rand() < 0.5:
                        actions = oracle_actions.copy()
                    else:
                        actions = agent_actions.copy()
                else:
                    actions = agent_actions.copy()
                    actions += np.random.randn(*actions.shape) * oracle.noise

            next_obs, rewards, dones, _, info = envs.step(actions)
            next_obs = next_obs.transpose(0, 3, 1, 2)

            # Assign obs for next loop iteration
            obs = next_obs.copy() 
            next_obs = next_obs.copy()

            # Modify next_obs if the episode is done to put the correct terminal observation
            for i in range(len(obs)):
                if dones[i]:
                    next_obs[i] = info[i]['terminal_observation'].transpose(2, 0, 1)

            transition_dict['actions'] = actions.copy()
            transition_dict['rewards'] = rewards.copy()
            transition_dict['next_obs'] = next_obs.copy()
            transition_dict['new_positions'] = envs.positions.copy()

            # Could become kinda big, so we delete it
            del next_obs, fovea_image, oracle_pos_after_saccade, oracle_actions, oracle_submoves, actions, rewards, dones, info

            buffer.add(transition_dict)

            # Sanity check on the buffer
            if buffer.is_full and not checked_buffer:
                checked_buffer = True
                batch = buffer.sample(10, batch_inds=np.arange(10)) 
                os.makedirs(savepath + 'buffer_test/', exist_ok=True)
                for i in range(10):
                    obs1 = batch['obs'][i].cpu().numpy()
                    obs2 = batch['next_obs'][i].cpu().numpy()
                    endpoints = batch['endpoints'][i].cpu().numpy()
                    position = batch['positions'][i].cpu().numpy()
                    new_position = batch['new_positions'][i].cpu().numpy()
                    n_symbols = batch['n_symbols'][i].cpu().numpy()
                    action = batch['actions'][i].cpu().numpy()
                    reward = batch['rewards'][i].cpu().numpy()
                    oracle_action = batch['oracle_actions'][i].cpu().numpy()
                    barycenters = batch['barycenters'][i].cpu().numpy()
                    oracle_saccade = batch['oracle_saccades'][i].cpu().numpy()
                    symbols_done = batch['symbols_done'][i].cpu().numpy()
                    fovea_image = batch['fovea_after_saccade'][i].cpu().numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

                    # Initial image
                    axes[0].imshow(obs1.transpose(2, 1, 0), extent=[-1,1,-1,1], origin='lower')
                    axes[0].plot([position[0], position[0] + action[0]], [position[1], position[1] + action[1]],  lw=8, color='gray', label='Total action')
                    axes[0].scatter(position[0] + oracle_saccade[0], position[1] + oracle_saccade[1], color='r', marker='.', label='Eye pos after saccade')
                    axes[0].legend()
                    axes[0].axvline(x=0., color='magenta', ls='--', lw=1)
                    axes[0].axhline(y=0., color='magenta', ls='--', lw=1)
                    axes[0].set_title(f"Initial obs, pos=({position[0]:.2f}|{position[1]:.2f})")

                    axes[1].imshow(fovea_image.transpose(2, 1, 0), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
                    axes[1].scatter(0, 0, color='r', marker='.', s=144, label='Eye pos')
                    axes[1].scatter(action[0] - oracle_saccade[0], action[1] - oracle_saccade[1], s=144, color='gray', marker='+', label='Hand pos')
                    axes[1].legend()
                    axes[1].set_title('Fovea image')

                    axes[2].imshow(obs2.transpose(2, 1, 0), extent=[-1,1,-1,1], origin='lower')
                    axes[2].axvline(x=0., color='magenta', ls='--', lw=1)
                    axes[2].axhline(y=0., color='magenta', ls='--', lw=1)
                    for barycenter, end, _ in zip(barycenters, endpoints, range(n_symbols)):
                        axes[2].plot([end[0], end[2]], [end[1], end[3]], lw=6, label=f"({end[0]:.2f}|{end[1]:.2f}) -> ({end[2]:.2f}|{end[3]:.2f})")
                        axes[2].scatter(barycenter[0], barycenter[1], s=100, marker='+', color='gray', label=f"Barycenter: ({barycenter[0]:.2f}|{barycenter[1]:.2f})")

                    axes[2].set_title(f"Final obs, pos=({new_position[0]:.2f}|{new_position[1]:.2f})")
                    axes[2].axvline(x=0., color='magenta', ls='--', lw=1)
                    axes[2].axhline(y=0., color='magenta', ls='--', lw=1)
                    axes[2].set_xlim(-1, 1)
                    axes[2].set_ylim(-1, 1)
                    box = axes[2].get_position()
                    axes[2].set_position([box.x0, box.y0, box.width * 0.8, box.height])

                    # Put a legend to the right of the current axis
                    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

                    fig.suptitle(f"Action: ({action[0]:.2f}|{action[1]:.2f}|{action[2]:.2f}), reward: {reward}, n_symbols: {n_symbols}, dones = {[int(d) for d in symbols_done[:n_symbols]]} \n")
                    fig.tight_layout()
                    fig.savefig(savepath + 'buffer_test/' + f"{i}.png")
                    plt.close(fig)

                    del obs1, obs2, endpoints, position, new_position, n_symbols, action, reward, oracle_action, barycenters, oracle_saccade, symbols_done, fovea_image

        # Start updating only once the buffer is full
        if not buffer.is_full:
            continue

        ###########################################################
        #################                         #################
        #################      Update phase       #################
        #################                         #################
        ###########################################################
        saccade_losses_buffer = []
        homing_losses_buffer = []
        for gd_step in range(args.training_params['gd_steps_per_rollout']):
            # Sample a batch from the buffer
            batch = buffer.sample(args.training_params['batch_size'])

            peripheral_obs = batch['obs']
            fovea_obs = batch['fovea_after_saccade']

            target_actions = batch['oracle_actions']
            target_saccades = batch['oracle_saccades']


            pred_saccades, pred_homings = agent(peripheral_obs, fovea_obs)
            

            if args.agent_params['fovea_ablated']:
                pred_actions = pred_saccades
                saccade_loss = tch.nn.functional.mse_loss(pred_actions, target_actions, reduction='none')
                homing_loss = saccade_loss.detach().clone()
            else:
                pred_actions = pred_saccades + pred_homings
                saccade_loss = tch.nn.functional.mse_loss(pred_saccades, tch.cat([target_saccades, tch.zeros((pred_saccades.shape[0], 1), dtype=tch.float, device=agent.device)], dim=1), reduction='none')
                homing_loss = tch.nn.functional.mse_loss(pred_actions, target_actions, reduction='none')

            saccade_losses_buffer.extend(saccade_loss.detach().cpu().numpy())
            homing_losses_buffer.extend(homing_loss.detach().cpu().numpy())

            saccade_loss = tch.mean(saccade_loss)
            homing_loss = tch.mean(homing_loss)

            # Backpropagate
            if not args.agent_params['fovea_ablated']:
                foveal_optimizer.zero_grad()
                homing_loss.backward(retain_graph=True)
                foveal_optimizer.step()

            peripheral_optimizer.zero_grad()
            saccade_loss.backward()
            peripheral_optimizer.step()
            writer.add_scalar('losses/saccade_loss', saccade_loss.item(), global_step=gd_step+ args.training_params['gd_steps_per_rollout'] * update)
            writer.add_scalar('losses/homing_loss', homing_loss.item(), global_step=gd_step+ args.training_params['gd_steps_per_rollout'] * update)

        saccade_losses_buffer = np.array(saccade_losses_buffer).flatten()
        homing_losses_buffer = np.array(homing_losses_buffer).flatten()

        logging.critical(f"Seed[{seed}] Update {update}:  error saccade = {np.mean(np.sqrt(saccade_losses_buffer)):.2e} pm {np.std(np.sqrt(saccade_losses_buffer)):.2e}, error homing = {np.mean(np.sqrt(homing_losses_buffer)):.2e} pm {np.std(np.sqrt(homing_losses_buffer)):.2e}")


        ###########################################################
        #################                         #################
        #################      Testing phase      #################
        #################                         #################
        ###########################################################

        if update % args.training_params['test_every'] == 0:
            logging.critical(f"Testing at update {update}")
            # Plot several sanity checks.
            os.makedirs(savepath + f"{update}/", exist_ok=True)
            tch.save(agent.state_dict(), savepath + f"{update}/agent.pt")

            # For these tests, we use the last batch used in GD steps

            # This is to make sure we don't have outliers in the MSE
            fig, axes = plt.subplots(2, 2, figsize=(20, 10))
            axes[0,0].hist(saccade_losses_buffer, bins=100, log=True)
            axes[0,0].set_title('Saccade errors histogram')
            if not args.agent_params['fovea_ablated']:
                axes[0,1].hist(homing_losses_buffer, bins=100, log=True)
                axes[0,1].set_title('Homing errors histogram')
            else:
                axes[0,1].set_title('Homing undefined for fovea ablated agent')
            axes[1,0].hist(np.log10(saccade_losses_buffer), bins=100, log=True)
            axes[1,0].set_title('Saccade log-errors histogram')
            if not args.agent_params['fovea_ablated']:
                axes[1,1].hist(np.log10(homing_losses_buffer), bins=100, log=True)
                axes[1,1].set_title('Homing log-errors histogram')
            else:
                axes[1,1].set_title('Homing undefined for fovea ablated agent')
            fig.tight_layout()
            fig.savefig(savepath + f"{update}/errors_histogram.png")

            for b_id in range(10):
                p_obs = peripheral_obs[b_id].cpu().numpy()
                f_obs = fovea_obs[b_id].cpu().numpy()
                target_a = target_actions[b_id].cpu().numpy()
                target_s = target_saccades[b_id].cpu().numpy()
                target_h = target_a.copy()
                target_h[:2] = target_h[:2] - target_s
                pred_s = pred_saccades[b_id].detach().cpu().numpy()
                pred_h = pred_homings[b_id].detach().cpu().numpy()
                start_pos = batch['positions'][b_id].cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(30, 15))
                axes[0].imshow(p_obs.transpose(2, 1, 0), extent=[-1, 1, -1, 1], origin='lower')
                # axes[0].plot([start_pos[0], start_pos[0]+target_s[0]], [start_pos[1], start_pos[1]+target_s[1]], s=144, marker='+', color='red', label='Target saccade')
                # axes[0].plot([start_pos[0], start_pos[0]+pred_s[0]], [start_pos[1], start_pos[1]+pred_s[1]], s=144, marker='x', color='green', label='Predicted saccade')
                axes[0].plot([start_pos[0], start_pos[0]+target_s[0]], [start_pos[1], start_pos[1]+target_s[1]], lw=12, color='red', label='Target saccade')
                axes[0].scatter(start_pos[0]+target_s[0], start_pos[1]+target_s[1], s=4096, marker='+', color='darkred')
                axes[0].plot([start_pos[0], start_pos[0]+pred_s[0]], [start_pos[1], start_pos[1]+pred_s[1]], lw=12, color='green', label='Predicted saccade')
                axes[0].scatter(start_pos[0]+pred_s[0], start_pos[1]+pred_s[1], s=4096, marker='x', color='darkgreen')
                axes[0].set_title('Peripheral vision system')
                axes[0].legend()

                axes[1].imshow(f_obs.transpose(2, 1, 0), extent=[-1/4, 1/4, -1/4, 1/4], origin='lower')
                axes[1].plot([0., pred_h[0]], [0., pred_h[1]], lw=12, color='green', label='Predicted homing')
                axes[1].scatter(pred_h[0], pred_h[1], marker='x', s=4096, color='darkgreen')
                axes[1].set_title('Foveal vision system')
                axes[1].legend()
                fig.tight_layout()
                fig.savefig(savepath + f"{update}/transition_{b_id}.png")
                plt.close(fig)


            # Using a newly generated rollout, follow fully the agent trajectory and do same diagnostics as for oracle.
            # If those trainings do not work, maybe add regularization losses for hand position etc... 
            test_obs = test_envs.reset().transpose(0, 3, 1, 2)
            test_done = np.zeros(test_envs.n_envs)
            for step in range(0, 20):
                test_saccades = agent.get_saccade(tch.from_numpy(test_obs).float().to(agent.device)).detach().cpu().numpy()
                test_pos_after_saccade = (test_envs.positions + test_saccades[:, :2]).copy()
                test_pos_after_saccade = np.clip(test_pos_after_saccade, -1, 1)

                # Just in case, and also to make it explicit we call get_center_patch with float position
                test_fovea_image = test_envs.get_centered_patches(center_pos=test_pos_after_saccade)
                test_fovea_image = tch.Tensor(np.array([im.transpose(2, 0, 1) for im in test_fovea_image])).float().to(agent.device)
                test_homings = agent.get_homing(test_fovea_image).detach().cpu().numpy()

                test_action = test_saccades + test_homings

                for env_idx in range(5):
                    fig, axes = plt.subplots(1, 2, figsize=(32, 16))
                    axes[0].imshow(test_obs[env_idx].transpose(2, 1, 0), origin='lower', extent=[-1, 1, -1, 1])
                    axes[0].plot([test_envs.positions[env_idx, 0], test_envs.positions[env_idx, 0] + test_action[env_idx, 0]], [test_envs.positions[env_idx, 1], test_envs.positions[env_idx, 1] + test_action[env_idx, 1]], lw=8, color='gray', label='Total action')
                    axes[0].scatter(test_pos_after_saccade[env_idx, 0], test_pos_after_saccade[env_idx, 1], color='m', marker='+', label='Initial saccade')
                    axes[0].legend()
                    axes[0].set_title('Global image')

                    # Plot the fovea image and homing submove 
                    if not args.agent_params['fovea_ablated']:
                        axes[1].imshow(test_fovea_image[env_idx].cpu().numpy().transpose(2, 1, 0), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
                        axes[1].scatter([test_homings[env_idx, 0]], [test_homings[env_idx, 1]], s=4096, color='m', marker='+', label='Homing submove')
                    else:
                        axes[1].imshow(np.zeros((64,64,3)), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])

                    axes[1].scatter(0, 0, color='r', marker='.', s=4096, label='Eye pos after saccade')
                    axes[1].legend()
                    axes[1].set_title('Fovea image')

                    fig.savefig(savepath + f'{update}/env_{env_idx}_step_{step}.png')
                    plt.close(fig)

                test_obs, test_reward, test_done, _, test_info = test_envs.step(test_action, test_done)
                test_obs = test_obs.transpose(0, 3, 1, 2)

                for env_idx in range(5):
                    if test_done[env_idx]:
                        fig, axes = plt.subplots(1, 2, figsize=(32, 16))
                        axes[0].imshow(test_info[env_idx]['terminal_observation'].transpose(1, 0, 2), origin='lower', extent=[-1, 1, -1, 1])
                        axes[0].set_title('Final observation')

                        # Plot the fovea image and homing submove 
                        axes[1].imshow(np.zeros((32, 32, 3)), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
                        axes[1].set_title('Placeholder fovea image')

                        fig.savefig(savepath + f'{update}/env_{env_idx}_step_{step}_final_observation.png')
                        plt.close(fig)
                
            # Delete all tests variables to avoid any unintended side-effect
            del test_reward, test_done, test_info, test_homings, test_saccades, test_action, test_pos_after_saccade, test_fovea_image

    tch.save(agent.state_dict(), savepath + f"/final_agent.pt")

def run_one_seed(seed):
    args = {
        'seed': seed,
        'root_output_folder': '/scratch/atf6569/saccade_drawing/initial_exps/',
        # 'root_output_folder': '~/Scratch/atf6569/saccade_drawing/initial_exps/',
        'run_name': 'test_run_',

        'board_params': {
            'n_envs': 32,
            'n_symbols_min': 2,
            'n_symbols_max': 3,
            'reward_type': 'default',
            'reward_params':  {'overlap_criterion': .2},
            'ordering_sensitivity': 0, # in pixels
            # 'all_envs_start_identical': True, 
            'all_envs_start_identical': False, 
            # Ensures we see different trajectories for each realization of lines, will be mixed in buffer so no "single env minibatches"
        },

        'oracle_params': {
            'sensitivity': .1,
            'noise': .02,
        },

        'agent_params': {
            'fovea_ablated': False,
            'peripheral_net_params': {
                'n_pixels_in': 128,
                'cnn_n_featuremaps': [128, 128, 128, 128, 128],
                'cnn_kernel_sizes': [5, 5, 3, 3, 3],
                'cnn_kernel_strides': [1, 1, 1, 1, 1],
                'fc_sizes': [1024, 1024],
                'rnn_size': 1024,
                # 'recurrence_type': 'rnn',
                # 'recurrent_steps': 1, # 1 step is equivalent to rec layer being feedforward, with same number of parameters !
                'recurrence_type': 'gru',
                'recurrent_steps': 5, # 1 step is equivalent to rec layer being feedforward, with same number of parameters !
                },
        
            'foveal_net_params': {
                'n_pixels_in': 32,
                'cnn_n_featuremaps': [128, 128, 128],
                'cnn_kernel_sizes': [5, 5, 3,],
                'cnn_kernel_strides': [1, 1, 1,],
                'fc_sizes': [512, 512],
                'rnn_size': 512,
                'recurrence_type': 'rnn',
                'recurrent_steps': 1, # 1 step is equivalent to rec layer being feedforward, with same number of parameters !
                },
        },

        'buffer_params': {
            'buffer_size': 5_000,
        },

        'training_params': {
            'n_updates': 10_000,
            'test_every': 500,
            'lr': 5e-4,
            # 'seed': seed,
            'batch_size': 64,
            # How many env steps we do collecting data before updating the agent
            'rollout_length': 10,
            # How many batches / GD steps we do after each rollout phase 
            'gd_steps_per_rollout': 5, # about rollout steps * n_envs / batch_size
            'start_agent_only_after': 2_500,
        },
    }


    
    peripheral_net_params = {
        'n_pixels_in': 128,
        'cnn_n_featuremaps': [64, 64, 64, 32, 32],
        'cnn_kernel_sizes': [5, 5, 3, 3, 3],
        'cnn_kernel_strides': [1, 1, 1, 1, 1],
        'fc_sizes': [256, 256],
        'rnn_size': 256,
        'recurrence_type': 'rnn',
        'recurrent_steps': 1, 
        }
    
    foveal_net_params = {
        'n_pixels_in': 32,
        'cnn_n_featuremaps': [128, 64, 64],
        'cnn_kernel_sizes': [5, 5, 3,],
        'cnn_kernel_strides': [1, 1, 1,],
        'fc_sizes': [512, 512],
        'rnn_size': 512,
        'recurrence_type': 'rnn',
        'recurrent_steps': 1, 
        }
    
 

    # MAybe it will reduce the number of edge cases ?
    net_names = ['default']
    peripheral_net_params = [peripheral_net_params]
    foveal_net_params = [foveal_net_params]

    for net_name, peripheral_net_param, foveal_net_param in zip(net_names, peripheral_net_params, foveal_net_params):
        # for all_envs_start_identical in [True, False]:
        for all_envs_start_identical in [False]:
            modified_args = deepcopy(args)    
            modified_args['agent_params']['peripheral_net_params'] = peripheral_net_param
            modified_args['agent_params']['foveal_net_params'] = foveal_net_param
            # modified_args['training_params']['run_name'] = f'{net_name}_envs_identical_{all_envs_start_identical}'
            modified_args['run_name'] = f'{net_name}_envs_identical_{all_envs_start_identical}'
            modified_args['board_params']['all_envs_start_identical'] = all_envs_start_identical
            train(modified_args)


if __name__ == "__main__":
    from multiprocessing import Pool
    n=4
    pool = Pool(n)
    pool.map(run_one_seed, range(n))