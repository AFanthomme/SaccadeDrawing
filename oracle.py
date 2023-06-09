import numpy as np
from copy import deepcopy

class RuleOracle:
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


                if envs.rules[env_idx] in ['rightward', 'leftward', 'upward', 'downward', 'leftward_with_another_color']:
                    next_symbol_idx = np.argmin(envs.symbols_done[env_idx])
                    # Careful, need to convert to [-1, 1] range
                    barycenter = envs.barycenters[env_idx, next_symbol_idx]

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

                elif envs.rules[env_idx] == 'closest':
                    # print('In closest branch')
                    min_dist = np.inf
                    saccade = np.zeros(2)
                    move = np.zeros(2)
                    next_symbol_idx = -1
                    start_from = None
                    draw = False
                    # action = np.concatenate([self.sample_noise(), -0.2*np.ones(1)], axis=0)

                    for symbol_idx in range(envs.n_symbols[env_idx]):
                        # First, determine what would be the closest ENDPOINT to move to
                        if envs.symbols_done[env_idx, symbol_idx] == 0:
                            # print(f'Symbol idx {symbol_idx} is not done')
                            barycenter = envs.barycenters[env_idx, symbol_idx]
                            start_point, end_point = envs.target_endpoints[env_idx, symbol_idx, :2] , envs.target_endpoints[env_idx, symbol_idx, 2:]
                            start_move = start_point - positions[env_idx]
                            end_move = end_point - positions[env_idx]


                            if np.linalg.norm(start_move) < min_dist:

                                min_dist = np.linalg.norm(start_move)
                                saccade = barycenter - positions[env_idx] 
                                move = deepcopy(start_move)
                                next_symbol_idx = symbol_idx
                                start_from = 'start'

                            if np.linalg.norm(end_move) < min_dist:

                                min_dist = np.linalg.norm(end_move)
                                saccade = barycenter - positions[env_idx] 
                                move = deepcopy(end_move)
                                next_symbol_idx = symbol_idx
                                start_from = 'end'
                            
                    # Then, see if it is closer than sensitivity, in which case need to change the move
                    if next_symbol_idx >= 0:
                        # Ensures we found a line to draw 
                        if min_dist < self.sensitivity:
                            # If move to start is smaller than sensitivity, we can directly move to the end point
                            barycenter = envs.barycenters[env_idx, next_symbol_idx]
                            start_point, end_point = envs.target_endpoints[env_idx, next_symbol_idx, :2] , envs.target_endpoints[env_idx, next_symbol_idx, 2:]
                            if start_from == 'start':
                                move = end_point - positions[env_idx]
                            elif start_from == 'end':
                                move = start_point - positions[env_idx]
                            saccade = barycenter - positions[env_idx] 
                            draw = True
                        
                    # Finally, put this into an action 
                    move = move + self.sample_noise()
                    actions[env_idx] = [move[0], move[1], .2*(2.*float(draw)-1)]
                    saccades[env_idx] = saccade + self.sample_noise()

                elif envs.rules[env_idx] == 'closest_from_left':
                    # Do symbols by proximity, but always left to right within the symbol
                    min_dist = np.inf
                    next_symbol_idx = -1

                    for symbol_idx in range(envs.n_symbols[env_idx]):
                        # First, determine what would be the closest BARYCENTET to move to
                        if envs.symbols_done[env_idx, symbol_idx] == 0:
                            barycenter = envs.barycenters[env_idx, symbol_idx]
                            dist = np.linalg.norm(barycenter-positions[env_idx])
                            if dist < min_dist:
                                min_dist = dist
                                next_symbol_idx = symbol_idx 

                    saccade = envs.barycenters[env_idx, next_symbol_idx] - positions[env_idx]
                    start_point, end_point = envs.target_endpoints[env_idx, next_symbol_idx, :2] , envs.target_endpoints[env_idx, next_symbol_idx, 2:]
                    start_move = start_point - positions[env_idx]
                    end_move = end_point - positions[env_idx]

                    if np.linalg.norm(start_move) < self.sensitivity:
                        actions[env_idx] = [end_move[0], end_move[1], .2]
                    else:
                        actions[env_idx] = [start_move[0], start_move[1], -.2]

                    actions[env_idx, :2] = actions[env_idx, :2] + self.sample_noise()
                    saccades[env_idx] = saccade + self.sample_noise()

                elif envs.rules[env_idx] == 'leftward_closest_endpoints':
                    next_symbol_idx = np.argmin(envs.symbols_done[env_idx])
                    # Careful, need to convert to [-1, 1] range
                    barycenter = envs.barycenters[env_idx, next_symbol_idx]

                    # Eye movement to center of next symbol (idea: comes from "peripheral vision" / the whole screen)
                    saccades[env_idx] = barycenter - positions[env_idx] + self.sample_noise()

                    # Eye movement directly to next line endpoints 
                    start_point, end_point = envs.target_endpoints[env_idx, next_symbol_idx, :2] , envs.target_endpoints[env_idx, next_symbol_idx, 2:]
                    moves_to_start[env_idx] = start_point - positions[env_idx] + self.sample_noise()
                    moves_to_end[env_idx] = end_point - positions[env_idx] + self.sample_noise()

                    # Homing movement (after saccade, that's how you get to the endpoint)
                    homes_to_start[env_idx] = moves_to_start[env_idx] - saccades[env_idx] 
                    homes_to_end[env_idx] = moves_to_end[env_idx] - saccades[env_idx] 

                    closest_move = moves_to_start[env_idx] if np.linalg.norm(moves_to_start[env_idx]) < np.linalg.norm(moves_to_end[env_idx]) else moves_to_end[env_idx]
                    furthest_move = moves_to_start[env_idx] if np.linalg.norm(moves_to_start[env_idx]) > np.linalg.norm(moves_to_end[env_idx]) else moves_to_end[env_idx]
                    
                    if np.linalg.norm(closest_move) < self.sensitivity:
                        # If move to start is smaller than sensitivity, we can directly move to the end point
                        actions[env_idx] = [furthest_move[0], furthest_move[1], .2]
                    else:
                        # Otherwise, we move to the start point
                        actions[env_idx] = [closest_move[0], closest_move[1], -.2]


                else:
                    raise ValueError('Unknown rule: {}'.format(envs.rules[env_idx]))       
                
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

    def get_saccade(self, envs):
        _, submoves = self.get_action_and_submoves(envs)
        return submoves['saccades']
 
class RandomAgent:
    def __init__(self, amplitude_saccade=.5, amplitude_correction=.1, seed=777, **kwargs) -> None:
        self.amplitude_saccade = amplitude_saccade
        self.amplitude_correction = amplitude_correction
        self.np_random = np.random.RandomState(seed)

    def get_action_and_submoves(self, envs):
        n_envs = envs.n_envs
        saccades = self.amplitude_saccade * self.np_random.uniform(-1, 1, size=(n_envs, 3))
        saccades[:, 2] = 0
        corrections = self.amplitude_correction * self.np_random.uniform(-1, 1, size=(n_envs, 3))

        submoves = {'saccades': saccades,}
        actions = saccades + corrections
        return actions, submoves

    def get_saccade(self, envs):
        _, submoves = self.get_action_and_submoves(envs)
        return submoves['saccades']
 
if __name__ == '__main__':
    import os
    from env import RuleBoards
    import matplotlib.pyplot as plt
    savepath = 'out/rule_oracle_tests'
    os.makedirs(savepath, exist_ok=True)

    board_params = {
        'n_envs': 12,
        'n_symbols_min': 4,
        'n_symbols_max': 8,
        'reward_type': 'default',
        'reward_params':  {'overlap_criterion': .4},
        'all_envs_start_identical': False,
        'allowed_symbols': ['line_0', 'line_1', 'line_2', 'line_3'],
        # 'allowed_rules': ['rightward', 'leftward', 'upward', 'downward'],
        # 'allowed_rules': ['closest'],
        'allowed_rules': ['leftward_with_another_color', 'leftward_closest_endpoints', 'closest_from_left'],
        'timeout': None,
        'mirror': False,
    }
    envs = RuleBoards(board_params, 777)
    oracle = RuleOracle(sensitivity=.05, noise=.01, seed=777)

    # Test the oracle in an open loop setting
    t_tot = 40

    obs = envs.reset()
    done = np.zeros(envs.n_envs)

    for step in range(0, t_tot):
        action, submoves = oracle.get_action_and_submoves(envs)
        saccades = submoves['saccades']
        pos_after_saccade = (envs.positions + saccades).copy()

        # Just in case, and also to make it explicit we call get_center_patch with float position
        pos_after_saccade = np.clip(pos_after_saccade, -1, 1)
        fovea_image = envs.get_centered_patches(center_pos=pos_after_saccade)
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

