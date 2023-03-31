import torch  as tch 
import numpy as np
import matplotlib.pyplot as plt

import os


class Buffer:
    def __init__(self, buffer_size=500, device='cuda'):
        self.buffer_size = buffer_size
        self.buffer_obs = np.zeros((buffer_size, 3, 128, 128), dtype=np.float32)
        self.buffer_actions = np.zeros((buffer_size, 3), dtype=np.float32)
        self.buffer_rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.buffer_next_obs = np.zeros((buffer_size, 3, 128, 128), dtype=np.float32)
        self.buffer_endpoints = np.zeros((buffer_size, 18, 4), dtype=np.float32)
        self.buffer_barycenters = np.zeros((buffer_size, 18, 2), dtype=np.float32)
        self.buffer_positions = np.zeros((buffer_size, 2), dtype=np.float32)
        self.buffer_new_positions = np.zeros((buffer_size, 2), dtype=np.float32)
        self.buffer_symbols_done = np.zeros((buffer_size, 5), dtype=np.int)
        self.buffer_n_symbols = np.zeros((buffer_size), dtype=int)
        # These two are all that matters: the total action, and the part of it that was a visual saccade
        self.buffer_oracle_actions = np.zeros((buffer_size, 3), dtype=np.float32)
        self.buffer_oracle_saccades = np.zeros((buffer_size, 2), dtype=np.float32)
        self.buffer_fovea_after_saccade = np.zeros((buffer_size, 3, 32, 32), dtype=np.float32)
        self.current_size = 0
        self.pos = 0
        self.is_full = False 
        self.device = tch.device(device)

    def _compute_scaler(self):
        self.m = np.mean(self.buffer_obs, axis=(0, 2, 3), keepdims=True)
        self.s = np.std(self.buffer_obs, axis=(0, 2, 3), keepdims=True)


    def normalize(self, obs):
        if isinstance(obs, tch.Tensor):
            return (obs - tch.from_numpy(self.m).to(self.device)) / tch.from_numpy(self.s).to(self.device)
        elif isinstance(obs, np.ndarray): 
            return (obs - self.m) / self.s
        else:
            raise ValueError(f'obs should be either a numpy array or a torch  tensor, not {type(obs)}')


    def add(self, transition_dict):
        n_envs = transition_dict['obs'].shape[0]
        assert transition_dict['obs'].shape[1] == 3, 'Obs should be channel-first, ie (n_envs, 3, 64, 64)'
        assert transition_dict['next_obs'].shape[1] == 3, 'Next_obs should be channel-first, ie (n_envs, 3, 64, 64)'
        for i in range(n_envs):
            self.buffer_obs[self.pos] = transition_dict['obs'][i].copy()
            self.buffer_actions[self.pos] = transition_dict['actions'][i].copy()
            self.buffer_rewards[self.pos] = transition_dict['rewards'][i].copy()
            self.buffer_next_obs[self.pos] = transition_dict['next_obs'][i].copy()
            self.buffer_endpoints[self.pos] = transition_dict['endpoints'][i].copy()
            self.buffer_barycenters[self.pos] = transition_dict['barycenters'][i].copy()
            self.buffer_positions[self.pos] = transition_dict['positions'][i].copy()
            self.buffer_symbols_done[self.pos] = transition_dict['symbols_done'][i].copy()
            self.buffer_new_positions[self.pos] = transition_dict['new_positions'][i].copy()
            self.buffer_oracle_actions[self.pos] = transition_dict['oracle_actions'][i].copy()
            self.buffer_oracle_saccades[self.pos] = transition_dict['oracle_saccades'][i].copy()
            self.buffer_fovea_after_saccade[self.pos] = transition_dict['fovea_after_saccade'][i].copy()
            
            # self.buffer_n_symbols[self.pos, transition_dict['n_symbols'][i]-1] = 1
            self.buffer_n_symbols[self.pos] = transition_dict['n_symbols'][i] - 1 # 0 means 1 line, 1 means 2 lines, etc.

            self.current_size = min(self.current_size+1, self.buffer_size)
            if self.current_size == self.buffer_size and not self.is_full:
                self.is_full = True
                self._compute_scaler()

            self.pos = (self.pos+1) % self.buffer_size

    def sample(self, batch_size):
        # Sample takes care of converting to pytch  tensors
        batch_inds = np.random.randint(0, self.current_size, size=batch_size)
        return_dict = {}
        return_dict['obs'] = tch .tensor(self.buffer_obs[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['actions'] = tch .tensor(self.buffer_actions[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['rewards'] = tch .tensor(self.buffer_rewards[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['n_symbols'] = tch .tensor(self.buffer_n_symbols[batch_inds], dtype=tch .long).to(self.device).clone()
        return_dict['next_obs'] = tch .tensor(self.buffer_next_obs[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['endpoints'] = tch .tensor(self.buffer_endpoints[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['barycenters'] = tch .tensor(self.buffer_barycenters[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['positions'] = tch .tensor(self.buffer_positions[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['symbols_done'] = tch .tensor(self.buffer_symbols_done[batch_inds], dtype=tch .long).to(self.device).clone()
        return_dict['new_positions'] = tch .tensor(self.buffer_new_positions[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['oracle_actions'] = tch .tensor(self.buffer_oracle_actions[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['oracle_saccades'] = tch .tensor(self.buffer_oracle_saccades[batch_inds], dtype=tch .float32).to(self.device).clone()
        return_dict['fovea_after_saccade'] = tch .tensor(self.buffer_fovea_after_saccade[batch_inds], dtype=tch .float32).to(self.device).clone()
        return return_dict

    def save(self, path):
        np.savez(path, obs=self.buffer_obs, actions=self.buffer_actions, rewards=self.buffer_rewards, next_obs=self.buffer_next_obs, 
                 endpoints=self.buffer_endpoints, positions=self.buffer_positions, symbols_done=self.buffer_symbols_done, new_positions=self.buffer_new_positions, 
                 oracle_actions=self.buffer_oracle_actions, n_symbols=self.buffer_n_symbols, m=self.m, s=self.s, barycenters=self.buffer_barycenters, 
                 oracle_saccades=self.buffer_oracle_saccades, fovea_after_saccade=self.buffer_fovea_after_saccade)

    def load(self, path):
        self.buffer_obs = np.load(path)['obs']
        self.buffer_actions = np.load(path)['actions']
        self.buffer_rewards = np.load(path)['rewards']
        self.buffer_next_obs = np.load(path)['next_obs']
        self.buffer_endpoints = np.load(path)['endpoints']
        self.buffer_positions = np.load(path)['positions']
        self.buffer_barycenters = np.load(path)['barycenters']
        self.buffer_symbols_done = np.load(path)['symbols_done']
        self.buffer_new_positions = np.load(path)['new_positions']
        self.buffer_oracle_actions = np.load(path)['oracle_actions']
        self.buffer_oracle_saccades = np.load(path)['oracle_saccades']
        self.buffer_fovea_after_saccade = np.load(path)['fovea_after_saccade']
        self.buffer_n_symbols = np.load(path)['n_symbols']
        self.m = np.load(path)['m']
        self.s = np.load(path)['s']
        self.is_full = True # To avoid recomputing the means if we do only partial loading


if __name__ == '__main__':
    import inspect
    from env import Boards, Oracle
    from copy import deepcopy


    seed = 777
    savepath = 'out/buffer_tests/'

    board_params = {
        'n_envs': 10,
        'n_symbols_min': 4,
        'n_symbols_max': 8,
        'reward_type': 'default',
        'reward_params': {},
        'all_envs_start_identical': False,
    }

    # Important: seeding
    np.random.seed(seed)
    tch .manual_seed(seed)
    tch .backends.cudnn.deterministic = True

    envs = Boards(board_params=board_params, seed=seed)
    oracle = Oracle(sensitivity=.1, noise=.02, seed=777)
    buffer = Buffer(10)

    obs = envs.reset().transpose(0, 3, 1, 2)
    for t in range(10):
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
        transition_dict['oracle_actions'] = oracle_actions.copy()
        transition_dict['oracle_saccades'] = oracle_submoves['saccades'].copy()

        # Just add some noise to make actions and oracle actions different in buffer
        actions = oracle_actions.copy() + np.random.normal(0, 0.02, size=actions.shape)


        # Just in case, and also to make it explicit we call get_center_patch with float position
        pos_after_saccade = (envs.positions + oracle_submoves['saccades']).copy()
        pos_after_saccade = np.clip(pos_after_saccade, -1, 1)
        fovea_image = np.array([envs.get_centered_patch(env_idx, center_pos=pos_after_saccade[env_idx]) for env_idx in range(envs.n_envs)])
        transition_dict['fovea_after_saccade'] = fovea_image.copy()


        next_obs, rewards, dones, _, info = envs.step(actions)
        next_obs = next_obs.transpose(0, 3, 1, 2)
        
        # Assign obs for next loop iteration
        obs = next_obs.copy() 

        # Modify next_obs if the episode is done to put the correct terminal observation
        for i in range(len(obs)):
            if dones[i]:
                next_obs[i] = info[i]['terminal_observation']

        transition_dict['actions'] = actions.copy()
        transition_dict['rewards'] = rewards.copy()
        transition_dict['next_obs'] = next_obs.copy()
        transition_dict['new_positions'] = envs.positions.copy()

        # Could become kinda big, so we delete it
        del next_obs

        buffer.add(transition_dict)

        if buffer.is_full:
            print(f'Buffer is full at step {t}')
            batch = buffer.sample(10) 
            os.makedirs(savepath, exist_ok=True)

            for i in range(10):
                obs1 = transition_dict['obs'][i]
                obs2 = transition_dict['next_obs'][i]
                endpoints = transition_dict['endpoints'][i]
                positions = transition_dict['positions'][i]
                new_positions = transition_dict['new_positions'][i]
                n_symbols = transition_dict['n_symbols'][i]
                actions = transition_dict['actions'][i]
                reward = transition_dict['rewards'][i]
                oracle_action = transition_dict['oracle_actions'][i]
                barycenters = transition_dict['barycenters'][i]
                oracle_saccades = transition_dict['oracle_saccades'][i]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Initial image
                axes[0].imshow(obs1.transpose(1, 2, 0), extent=[-1,1,-1,1], origin='lower')
                axes[0].plot([positions[i, 0], positions[i, 0] + actions[i, 0]], [positions[i, 1], positions[i, 1] + actions[i, 1]], color='gray', label='Total action')
                axes[2].scatter(barycenter[0], barycenter[1], s=100, marker='+', color='gray', label=f"Barycenter: ({barycenter[0]:.2f}|{barycenter[1]:.2f})")
                axes[0].plot([positions[i, 0], positions[i, 0] + oracle_saccades[i, 0]], [positions[i, 1], positions[i, 1] + oracle_saccades[i, 1]], color='gray', label='Total action')
                axes[0].legend()
                axes[0].axvline(x=0., color='magenta', ls='--', lw=4)
                axes[0].axhline(y=0., color='magenta', ls='--', lw=4)
                axes[0].set_title(f"Initial obs, pos=({positions[0]:.2f}|{positions[1]:.2f})")


                axes[1].imshow(obs2.transpose(1, 2, 0), extent=[-1,1,-1,1], origin='lower')
                axes[1].axvline(x=0., color='magenta', ls='--', lw=4)
                axes[1].axhline(y=0., color='magenta', ls='--', lw=4)
                axes[1].set_title(f"Final obs, pos=({new_positions[0]:.2f}|{new_positions[1]:.2f})")


                for barycenter, end in zip(barycenters, endpoints):
                    axes[2].plot([end[0], end[2]], [end[1], end[3]], lw=4, label=f"({end[0]:.2f}|{end[1]:.2f}) -> ({end[2]:.2f}|{end[3]:.2f})")
                axes[2].axvline(x=0., color='magenta', ls='--', lw=4)
                axes[2].axhline(y=0., color='magenta', ls='--', lw=4)
                axes[2].set_xlim(-1, 1)
                axes[2].set_ylim(-1, 1)
                axes[2].legend()



            # axes[0].imshow(obs[env_idx].transpose(1, 0, 2), origin='lower', extent=[-1, 1, -1, 1])
            # axes[0].plot([envs.positions[env_idx, 0], envs.positions[env_idx, 0] + action[env_idx, 0]], [envs.positions[env_idx, 1], envs.positions[env_idx, 1] + action[env_idx, 1]], color='gray', label='Total action')
            # axes[0].scatter([envs.positions[env_idx, 0] + saccades[env_idx, 0]], [envs.positions[env_idx, 1] +saccades[env_idx, 1]], color='m', marker='+', label='Initial saccade')
            # axes[0].legend()
            # axes[0].set_title('Global image')

            # # Plot the fovea image and homing submove 
            # # print(fovea_image[env_idx].shape)
            # axes[1].imshow(fovea_image[env_idx].transpose(1, 0, 2), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
            # axes[1].scatter([homing_start[env_idx, 0]], [homing_start[env_idx, 1]], color='m', marker='+', label='Identified start')
            # axes[1].scatter([homing_end[env_idx, 0]], [homing_end[env_idx, 1]], color='c', marker='+', label='Identified end')
            # axes[1].scatter(0, 0, color='r', marker='.', s=36, label='Eye pos after saccade')
            # axes[1].plot([homing_start[env_idx, 0], homing_end[env_idx, 0]], [homing_start[env_idx, 1], homing_end[env_idx, 1]], color='gray', marker='+')
            # axes[1].legend()
            # axes[1].set_title('Fovea image')

                fig.suptitle(f"Action: ({action[0]:.2f}|{action[1]:.2f}|{action[2]:.2f}), reward: {reward},  oracle action: ({oracle_action[0]:.2f}|{oracle_action[1]:.2f}|{oracle_action[2]:.2f}), n_symbols: {n_symbols} \n")
                plt.savefig(savepath + f"buffer_sanity_check/{i}_before.png")

            # for i in range(10):
            #     obs1 = batch['obs'][i].cpu().numpy()
            #     obs2 = batch['next_obs'][i].cpu().numpy()
            #     endpoints = batch['endpoints'][i]
            #     position = batch['positions'][i]
            #     new_position = batch['new_positions'][i]
            #     n_symbols = batch['n_symbols'][i]
            #     action = batch['actions'][i]
            #     reward = batch['rewards'][i]
            #     oracle_action = batch['oracle_actions'][i]

            #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            #     axes[0].imshow(obs1.transpose(1, 2, 0), extent=[-1,1,-1,1], origin='lower')
            #     axes[0].axvline(x=0., color='magenta', ls='--', lw=4)
            #     axes[0].axhline(y=0., color='magenta', ls='--', lw=4)
            #     axes[0].set_title(f"Initial obs, pos=({position[0]:.2f}|{position[1]:.2f})")
            #     axes[1].imshow(obs2.transpose(1, 2, 0), extent=[-1,1,-1,1], origin='lower')
            #     axes[1].axvline(x=0., color='magenta', ls='--', lw=4)
            #     axes[1].axhline(y=0., color='magenta', ls='--', lw=4)
            #     axes[1].set_title(f"Final obs, pos=({new_position[0]:.2f}|{new_position[1]:.2f})")


            #     for end in endpoints:
            #         end = end.cpu().numpy()
            #         axes[2].plot([end[0], end[2]], [end[1], end[3]], lw=4, label=f"({end[0]:.2f}|{end[1]:.2f}) -> ({end[2]:.2f}|{end[3]:.2f})")
            #     axes[2].axvline(x=0., color='magenta', ls='--', lw=4)
            #     axes[2].axhline(y=0., color='magenta', ls='--', lw=4)
            #     axes[2].set_xlim(-1, 1)
            #     axes[2].set_ylim(-1, 1)
            #     axes[2].legend()

            #     fig.suptitle(f"Action: ({action[0]:.2f}|{action[1]:.2f}), reward: {reward.item()},  oracle action: ({oracle_action[0]:.2f}|{oracle_action[1]:.2f}), n_symbols: {n_symbols+1} \n")
            #     plt.savefig(root_output_folder + f"/buffer_sanity_check/after_buffer_{i}.png")

            # Save and load the buffer to make sure it works


            buffer.save(savepath + 'test.npz')

            bkp_buffer = deepcopy(buffer)
            del buffer
            buffer = Buffer(10)
            buffer.load(savepath + 'test.npz')

            attributes = inspect.getmembers(buffer, lambda a: not(inspect.isroutine(a)))
            attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]

            for key, item in attributes:
                assert bkp_buffer.__dict__[key] == item, f'Buffer attribute {key} has changed when saving and loading'
