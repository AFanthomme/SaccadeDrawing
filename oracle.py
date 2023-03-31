import numpy as np

class Oracle:
    # Works on batched environments
    def __init__(self, sensitivity=.1, noise=.02, seed=777):
        self.sensitivity = sensitivity
        self.noise = noise
        self.np_random = np.random.RandomState(seed)

    @staticmethod
    def argsort(seq):
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__)
    
    def sample_noise(self):
        return self.np_random.normal(scale=self.noise, size=2)

    def get_action_and_submoves(self, envs):
        n_envs = envs.n_envs
        actions = np.zeros((n_envs, 3))
        positions = envs.positions.copy()
        moves_to_start = np.zeros((n_envs, 2))
        moves_to_end = np.zeros((n_envs, 2))
        saccades = np.zeros((n_envs, 2))
        homes_to_start = np.zeros((n_envs, 2))
        homes_to_end = np.zeros((n_envs, 2))

        for env_idx in range(n_envs):

            if np.all(envs.symbols_done[env_idx]):
                # If all lines are done, no need to continue drawing / moving
                actions[env_idx] = [0, 0, -0.1]
                continue
            else:
                # This returns the first index that has "done = 0" (done is 0 or 1, so min is 0 = not done) 
                next_symbol_idx = np.argmin(envs.symbols_done[env_idx])

                # Careful, need to convert to [-1, 1] range
                # barycenter = envs.barycenters[env_idx, next_symbol_idx] / 64. - 1.
                barycenter = envs.barycenters[env_idx, next_symbol_idx]
                print(barycenter)

                # Eye movement to center of next symbol (idea: comes from "peripheral vision" / the whole screen)
                saccades[env_idx] = barycenter - positions[env_idx] + self.sample_noise()

                # Eye movement directly to next line endpoints 
                # start_point, end_point = envs.target_endpoints[env_idx, next_symbol_idx, :2] / 64. - 1. , envs.target_endpoints[env_idx, next_symbol_idx, 2:] / 64. - 1.
                start_point, end_point = envs.target_endpoints[env_idx, next_symbol_idx, :2] , envs.target_endpoints[env_idx, next_symbol_idx, 2:]
                moves_to_start[env_idx] = start_point - positions[env_idx] + self.sample_noise()
                moves_to_end[env_idx] = end_point - positions[env_idx] + self.sample_noise()

                # Homing movement (after saccade, that's how you get to the endpoint)
                homes_to_start[env_idx] = moves_to_start[env_idx] - saccades[env_idx] 
                homes_to_end[env_idx] = moves_to_end[env_idx] - saccades[env_idx] 
                
                if np.linalg.norm(moves_to_start[env_idx]) < self.sensitivity:
                    # If move to start is smaller than sensitivity, we can directly move to the end point
                    actions[env_idx] = [moves_to_end[env_idx, 0], moves_to_end[env_idx, 1], .2]
                else:
                    # Otherwise, we move to the start point
                    actions[env_idx] = [moves_to_start[env_idx, 0], moves_to_start[env_idx, 1], -.2]
                
        submoves = {
            'saccades': saccades,
            'moves_to_start': moves_to_start,
            'moves_to_end': moves_to_end,
            'homes_to_start': homes_to_start,
            'homes_to_end': homes_to_end
            }

        return actions, submoves
    
    def get_action(self, envs):
        actions, _ = self.get_action_and_submoves(envs)
        return actions
 
 
if __name__ == '__main__':
    import os
    from env import Boards
    import matplotlib.pyplot as plt
    savepath = 'out/oracle_tests'
    os.makedirs(savepath, exist_ok=True)

    board_params = {
        'n_envs': 64,
        'n_symbols_min': 4,
        'n_symbols_max': 8,
        'reward_type': 'default',
        'reward_params': {'overlap_criterion': .4},
        'all_envs_start_identical': False,
    }

    envs = Boards(board_params, 777)
    oracle = Oracle(sensitivity=.1, noise=.02, seed=777)

    # Test the oracle in an open loop setting
    t_tot = 40


    obs = envs.reset()
    done = np.zeros(envs.n_envs)

    

    for step in range(0, t_tot):
        action, submoves = oracle.get_action_and_submoves(envs)
        saccades = submoves['saccades']
        pos_after_saccade = (envs.positions + saccades).copy()
        # print(pos_after_saccade)

        # Just in case, and also to make it explicit we call get_center_patch with float position
        pos_after_saccade = np.clip(pos_after_saccade, -1, 1)
        # print(pos_after_saccade)
        fovea_image = [envs.get_centered_patch(env_idx, center_pos=pos_after_saccade[env_idx]) for env_idx in range(envs.n_envs)]
        homing_start = submoves['homes_to_start']
        homing_end = submoves['homes_to_end']

        for env_idx in range(5):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(obs[env_idx].transpose(1, 0, 2), origin='lower', extent=[-1, 1, -1, 1])
            axes[0].plot([envs.positions[env_idx, 0], envs.positions[env_idx, 0] + action[env_idx, 0]], [envs.positions[env_idx, 1], envs.positions[env_idx, 1] + action[env_idx, 1]], color='gray', label='Total action')
            axes[0].scatter(pos_after_saccade[env_idx, 0], pos_after_saccade[env_idx, 1], color='m', marker='+', label='Initial saccade')
            axes[0].legend()
            axes[0].set_title('Global image')

            # Plot the fovea image and homing submove 
            # print(fovea_image[env_idx].shape)
            axes[1].imshow(fovea_image[env_idx].transpose(1, 0, 2), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
            axes[1].scatter([homing_start[env_idx, 0]], [homing_start[env_idx, 1]], color='m', marker='+', label='Identified start')
            axes[1].scatter([homing_end[env_idx, 0]], [homing_end[env_idx, 1]], color='c', marker='+', label='Identified end')
            axes[1].scatter(0, 0, color='r', marker='.', s=36, label='Eye pos after saccade')
            axes[1].plot([homing_start[env_idx, 0], homing_end[env_idx, 0]], [homing_start[env_idx, 1], homing_end[env_idx, 1]], color='gray', marker='+')
            axes[1].legend()
            axes[1].set_title('Fovea image')

            fig.savefig(savepath + f'/env_{env_idx}_step_{step}.png')
            plt.close(fig)

        obs, reward, done, truncated, info = envs.step(action, done)


        for env_idx in range(5):
            if done[env_idx]:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(info[env_idx]['terminal_observation'].transpose(1, 0, 2), origin='lower', extent=[-1, 1, -1, 1])
                axes[0].set_title('Final observation')

                # Plot the fovea image and homing submove 
                axes[1].imshow(np.zeros_like(fovea_image[env_idx]), origin='lower', extent=[-1/4, 1/4, -1/4, 1/4])
                axes[1].set_title('Placeholder fovea image')

                fig.savefig(savepath + f'/env_{env_idx}_step_{step}_final_observation.png')
                plt.close(fig)

