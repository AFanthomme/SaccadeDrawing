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
        self.buffer_symbols_done = np.zeros((buffer_size, 18), dtype=int)
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

    def sample(self, batch_size, batch_inds=None):
        # Sample takes care of converting to pytch  tensors
        if batch_inds is None:
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
                 oracle_saccades=self.buffer_oracle_saccades, fovea_after_saccade=self.buffer_fovea_after_saccade, current_size=self.current_size)

    def load(self, path):
        state_dict = np.load(path)
        self.buffer_obs = state_dict['obs']
        self.buffer_actions = state_dict['actions']
        self.buffer_rewards = state_dict['rewards']
        self.buffer_next_obs = state_dict['next_obs']
        self.buffer_endpoints = state_dict['endpoints']
        self.buffer_positions = state_dict['positions']
        self.buffer_barycenters = state_dict['barycenters']
        self.buffer_symbols_done = state_dict['symbols_done']
        self.buffer_new_positions = state_dict['new_positions']
        self.buffer_oracle_actions = state_dict['oracle_actions']
        self.buffer_oracle_saccades = state_dict['oracle_saccades']
        self.buffer_fovea_after_saccade = state_dict['fovea_after_saccade']
        self.buffer_n_symbols = state_dict['n_symbols']
        self.current_size = state_dict['current_size']
        self.m = state_dict['m']
        self.s = state_dict['s']
        self.is_full = True # To avoid recomputing the means if we do only partial loading


if __name__ == '__main__':
    import inspect
    from env import Boards
    from oracle import Oracle
    from copy import deepcopy


    seed = 777
    savepath = 'out/buffer_tests/'
    os.makedirs(savepath + 'before_buffer', exist_ok=True)

    board_params = {
        'n_envs': 10,
        'n_symbols_min': 4,
        'n_symbols_max': 8,
        'reward_type': 'default',
        'reward_params': {'overlap_criterion': .4},
        'all_envs_start_identical': False,
    }

    # Important: seeding
    np.random.seed(seed)
    tch.manual_seed(seed)
    tch.backends.cudnn.deterministic = True

    envs = Boards(board_params=board_params, seed=seed)
    oracle = Oracle(sensitivity=.1, noise=.02, seed=777)
    buffer = Buffer(10)

    obs = envs.reset().transpose(0, 3, 1, 2)
    for t in range(16):
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
        actions = oracle_actions.copy() + np.random.normal(0, 0.02, size=oracle_actions.shape)


        # Just in case, and also to make it explicit we call get_center_patch with float position
        pos_after_saccade = (envs.positions + oracle_submoves['saccades']).copy()
        pos_after_saccade = np.clip(pos_after_saccade, -1, 1)

        fovea_image = np.array([envs.get_centered_patch(env_idx, center_pos=pos_after_saccade[env_idx]).transpose(2, 0, 1) for env_idx in range(envs.n_envs)])
        transition_dict['fovea_after_saccade'] = fovea_image.copy()

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
        del next_obs, fovea_image, pos_after_saccade, oracle_actions, oracle_submoves, actions, rewards, dones, info

        buffer.add(transition_dict)

        if buffer.is_full:
            # Just to check it's full after one step !
            print(f'Buffer is full at step {t}')
            # We specify batch_inds to make sure we get the obs as they were put in
            batch = buffer.sample(10, batch_inds=np.arange(10)) 
            os.makedirs(savepath, exist_ok=True)

            for i in range(10):
                obs1 = transition_dict['obs'][i]
                obs2 = transition_dict['next_obs'][i]
                endpoints = transition_dict['endpoints'][i]
                position = transition_dict['positions'][i]
                new_position = transition_dict['new_positions'][i]
                n_symbols = transition_dict['n_symbols'][i]
                action = transition_dict['actions'][i]
                reward = transition_dict['rewards'][i]
                oracle_action = transition_dict['oracle_actions'][i]
                barycenters = transition_dict['barycenters'][i]
                oracle_saccade = transition_dict['oracle_saccades'][i]
                symbols_done = transition_dict['symbols_done'][i]
                fovea_image = transition_dict['fovea_after_saccade'][i]

                fig, axes = plt.subplots(1, 3, figsize=(30, 10))

                # Initial image
                axes[0].imshow(obs1.transpose(2, 1, 0), extent=[-1,1,-1,1], origin='lower')
                axes[0].plot([position[0], position[0] + action[0]], [position[1], position[1] + action[1]], color='gray', label='Total action')
                axes[0].scatter(position[0] + oracle_saccade[0], position[1] + oracle_saccade[1], color='r', marker='.', label='Eye pos after saccade')
                axes[0].legend()
                axes[0].axvline(x=0., color='magenta', ls='--', lw=1)
                axes[0].axhline(y=0., color='magenta', ls='--', lw=1)
                axes[0].set_title(f"Initial obs, pos=({position[0]:.2f}|{position[1]:.2f})")

                axes[1].imshow(fovea_image.transpose(2, 1, 0), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
                axes[1].scatter(0, 0, color='r', marker='.', s=36, label='Eye pos')
                axes[1].scatter(action[0] - oracle_saccade[0], action[1] - oracle_saccade[1], color='gray', marker='+', label='Hand pos')
                axes[1].legend()
                axes[1].set_title('Fovea image')

                axes[2].imshow(obs2.transpose(2, 1, 0), extent=[-1,1,-1,1], origin='lower')
                axes[2].axvline(x=0., color='magenta', ls='--', lw=1)
                axes[2].axhline(y=0., color='magenta', ls='--', lw=1)
                for barycenter, end, _ in zip(barycenters, endpoints, range(n_symbols)):
                    axes[2].plot([end[0], end[2]], [end[1], end[3]], lw=4, label=f"({end[0]:.2f}|{end[1]:.2f}) -> ({end[2]:.2f}|{end[3]:.2f})")
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
                fig.savefig(savepath + 'before_buffer/' + f"{i}_{t}.png")
                plt.close(fig)


            os.makedirs(savepath + 'after_buffer/', exist_ok=True)
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
                axes[0].plot([position[0], position[0] + action[0]], [position[1], position[1] + action[1]], color='gray', label='Total action')
                axes[0].scatter(position[0] + oracle_saccade[0], position[1] + oracle_saccade[1], color='r', marker='.', label='Eye pos after saccade')
                axes[0].legend()
                axes[0].axvline(x=0., color='magenta', ls='--', lw=1)
                axes[0].axhline(y=0., color='magenta', ls='--', lw=1)
                axes[0].set_title(f"Initial obs, pos=({position[0]:.2f}|{position[1]:.2f})")

                axes[1].imshow(fovea_image.transpose(2, 1, 0), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
                axes[1].scatter(0, 0, color='r', marker='.', s=36, label='Eye pos')
                axes[1].scatter(action[0] - oracle_saccade[0], action[1] - oracle_saccade[1], color='gray', marker='+', label='Hand pos')
                axes[1].legend()
                axes[1].set_title('Fovea image')

                axes[2].imshow(obs2.transpose(2, 1, 0), extent=[-1,1,-1,1], origin='lower')
                axes[2].axvline(x=0., color='magenta', ls='--', lw=1)
                axes[2].axhline(y=0., color='magenta', ls='--', lw=1)
                for barycenter, end, _ in zip(barycenters, endpoints, range(n_symbols)):
                    axes[2].plot([end[0], end[2]], [end[1], end[3]], lw=4, label=f"({end[0]:.2f}|{end[1]:.2f}) -> ({end[2]:.2f}|{end[3]:.2f})")
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
                fig.savefig(savepath + 'after_buffer/' + f"{i}_{t}.png")
                plt.close(fig)


            # Save and load the buffer to make sure it works


            buffer.save(savepath + 'test.npz')

            bkp_buffer = deepcopy(buffer)
            del buffer
            buffer = Buffer(10)
            buffer.load(savepath + 'test.npz')

            attributes = inspect.getmembers(buffer, lambda a: not(inspect.isroutine(a)))
            attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]

            for key, item in attributes:
                try:
                    # For arrays
                    assert (bkp_buffer.__dict__[key] == item).all(), f'Buffer attribute {key} has changed when saving and loading'
                except AttributeError:
                    # For the rest
                    assert bkp_buffer.__dict__[key] == item, f'Buffer attribute {key} has changed when saving and loading'