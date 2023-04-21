# Do we want to have sensitivity or not? Still running sims to know


import numpy as np
from copy import deepcopy
import logging
import torch as tch
import matplotlib
matplotlib.use('Agg')
from matplotlib import patches
import matplotlib.pyplot as plt
import os
import cv2

from typing import List, Tuple, Dict, Union, Optional, Any  


from functools import cmp_to_key, partial
from types import SimpleNamespace


# Keep the symbols outside of Board class for now, might change later

symbol_types = ['line_0', 'line_1', 'line_2', 'line_3',]

symbols_starts_in_bc_frame = {
    'line_0':  np.array([8, 16]),
    'line_1':  np.array([16, 8]),
    'line_2':  np.array([10, 10]),
    'line_3':  np.array([10, 22]),
}

symbols_ends_in_bc_frame = {
    'line_0':  np.array([24, 16]),
    'line_1':  np.array([16, 24]),
    'line_2':  np.array([22, 22]),
    'line_3':  np.array([22, 10]),
}

# Use always chennel last format for simplicity (h, w, 3)
# Transpose is here for the visual interpretability; doesn't matter for agent as long as positions are kept coherent.

types2patch = {
    'line_0':  cv2.line(np.zeros((32, 32, 3), dtype=np.uint8), (8, 16), (24, 16), (0,1,0), 2).transpose(1, 0, 2),
    'line_1':  cv2.line(np.zeros((32, 32, 3), dtype=np.uint8), (16, 8), (16, 24), (0,1,0), 2).transpose(1, 0, 2),
    'line_2':  cv2.line(np.zeros((32, 32, 3), dtype=np.uint8), (10, 10), (22, 22), (0,1,0), 2).transpose(1, 0, 2),
    'line_3':  cv2.line(np.zeros((32, 32, 3), dtype=np.uint8), (10, 22), (22, 10), (0,1,0), 2).transpose(1, 0, 2),
}



class Artist:
    def __init__(self, barycenter=(64, 64), symbol_type='line_0'):
        self.barycenter = np.array([barycenter[0],barycenter[1]]) # in pixels, in the global frame

        self.symbol_type = symbol_type
        self.start_in_bc_frame = symbols_starts_in_bc_frame[self.symbol_type]
        self.end_in_bc_frame = symbols_ends_in_bc_frame[self.symbol_type]
        self.start = self.start_in_bc_frame + self.barycenter - 16 # Barycenter is at position 16, 16
        self.end = self.end_in_bc_frame + self.barycenter - 16 # Barycenter is at position 16, 16

    def draw(self, board):
        x, y = self.barycenter
        patch = types2patch[self.symbol_type]
        board[x-16:x+16, y-16:y+16] = (board[x-16:x+16, y-16:y+16] + patch).clip(0,1)
        return board

# This will be for the "global state"   
class RuleBoards:
    def __init__(self, board_params, seed=0):
        self.seed = seed
        self.n_pixels = 128 # Hardcoded as we want a fancy setup.
        self.board_params = board_params 
        self.n_envs = board_params['n_envs']
        self.n_symbols_min = board_params['n_symbols_min']
        self.n_symbols_max = board_params['n_symbols_max']
        self.reward_type = board_params['reward_type']
        self.reward_params = SimpleNamespace(**board_params['reward_params'])
        self.all_envs_start_identical = board_params['all_envs_start_identical']
        self.allowed_symbols = board_params['allowed_symbols']
        self.allowed_rules = board_params['allowed_rules']

        # See figure for explanations
        # format is (xmin, xmax), (ymin, ymax)
        self.spawn_zones = [
            # First column
            ( (16, 32), (16, 32) ), 
            ( (16, 32), (48, 64) ),
            ( (16, 32), (80, 96) ),

            # Second column
            ( (32, 48), (32, 48) ),
            ( (32, 48), (64, 80) ),
            ( (32, 48), (96, 112) ),

            # Third column
            ( (48, 64), (16, 32) ),
            ( (48, 64), (48, 64) ),
            ( (48, 64), (80, 96) ),

            # Fourth column
            ( (64, 80), (32, 48) ),
            ( (64, 80), (64, 80) ),
            ( (64, 80), (96, 112) ),

            # Fifth column
            ( (80, 96), (16, 32) ),
            ( (80, 96), (48, 64) ),
            ( (80, 96), (80, 96) ),

            # Sixth column
            ( (96, 112), (32, 48) ),
            ( (96, 112), (64, 80) ),
            ( (96, 112), (96, 112) ),

        ]

        # Defines the rules 

        self.rules_names = ['closest', 'rightward', 'leftward', 'upward', 'downward']

        self.rule_border_colors = {
            'closest': np.array([0, 0, 0]),
            'leftward': np.array([.5, .5, 0]),
            'rightward': np.array([0.5, 0.5, 0.5]),
            'upward': np.array([0, .5, .5]),
            'downward': np.array([.5, 0, .5]),
        }

        self.backgrounds = {}

        for k,v in self.rule_border_colors.items():
            bkg = np.zeros((128, 128, 3), dtype=np.float32)
            bkg[0, 0] = 1
            tmp = np.tile(v, (128, 4, 1))
            bkg[:, :4, :] = tmp.copy()
            bkg[:, -4:, :] = tmp.copy()
            tmp = np.tile(v, (4, 128, 1))
            bkg[:4, :, :] = tmp.copy()
            bkg[-4:, :, :] = tmp.copy()
            self.backgrounds[k] = bkg.copy()

        self.default_pos_patch = np.zeros((128, 128, 3), dtype=np.uint8)
        self.default_pos_patch = cv2.circle(self.default_pos_patch, (64,64), 2, (1,0,0), -1)

        self.set_seed()
        self.reset()

    def set_seed(self):
        logging.critical(f'Called world.set_seed with seed {self.seed}')
        self.np_random = np.random.RandomState(seed=self.seed)

    @staticmethod
    def rule_defined_ordering(rule, b1, b2):
        # NOTE: 'closest' does not really use this ordering so use leftward just for the sake of it
        if rule == 'rightward' or rule == 'closest':
            if b1[0] != b2[0]:
                return b1[0]-b2[0]
            else:
                # Tie-breaker
                return b1[1]-b2[1] 
        elif rule == 'leftward':
            if b1[0] != b2[0]:
                return b2[0]-b1[0]
            else:
                # Tie-breaker, opposite to rightward for more variety
                return b2[1]-b1[1] 
        elif rule == 'upward':
            if b1[1] != b2[1]:
                return b1[1]-b2[1]
            else:
                # Tie-breaker
                return b1[0]-b2[0]
        elif rule == 'downward':
            if b1[1] != b2[1]:
                return b2[1]-b1[1]
            else:
                # Tie-breaker, opposite to upward for more variety
                return b2[0]-b1[0]

    def get_centered_patch(self, env_id: int, center_idx=None, center_pos=None) -> np.ndarray:
        assert not (center_idx is None and center_pos is None)
        if center_idx is None:
            # If center pos is a float, it's in [-1,1] and we need to convert it to pixels
            center_idx = np.array(((center_pos + 1.)/2.*self.n_pixels), dtype=int).clip(16, 112)
        x, y = center_idx
        return self.boards[env_id, x-16:x+16, y-16:y+16]

    def get_centered_patches(self, center_idx=None, center_pos=None) -> np.ndarray:
        if center_idx is None and center_pos is not None:
            return np.array([self.get_centered_patch(i, center_idx=None, center_pos=center_pos[i]) for i in range(self.n_envs)])
        elif center_pos is None and center_idx is not None:
            return np.array([self.get_centered_patch(i, center_idx=center_idx[i], center_pos=None) for i in range(self.n_envs)])
        else:
            raise ValueError('Exactly one of center and center_pos must be None')

    def _positions_to_patch(self, new_positions):
        positions_int = np.array(((new_positions + 1.)/2.*self.n_pixels), dtype=int)
        patches = np.zeros((self.n_envs, 128, 128, 3), dtype=np.uint8)
        for i, pos in enumerate(positions_int):
            patches[i] = cv2.circle(patches[i], tuple(pos), 2, (1,0,0), -1)
        return patches
    
    def _generate_one_board(self, rule_idx):
        # rule_name = self.rules_names[rule_idx]
        rule_name = self.allowed_rules[rule_idx]
        n_symbols = self.np_random.randint(self.n_symbols_min, self.n_symbols_max+1)
        symbols_done = np.zeros(18, dtype=bool) # Hardcode max, but might want to use lower
        symbols_done[n_symbols:] = True # Non existent symbols are still considered as symbols, just already done
        barycenters = np.zeros((18, 2), dtype=int)

        # First, draw the spawn zones
        spawn_ranges = self.np_random.permutation(self.spawn_zones)[:n_symbols]

        # Default
        # for i, (x_range, y_range) in enumerate(spawn_ranges):
        #     x = self.np_random.randint(x_range[0], x_range[1])
        #     y = self.np_random.randint(y_range[0], y_range[1])
        #     barycenters[i] = np.array([x, y])

        # New: remove any ambiguously ordered barycenters 
        # Careful as this might become slow if there are too many lines...
        for i, (x_range, y_range) in enumerate(spawn_ranges):
            loop = True
            while loop:
                x = self.np_random.randint(x_range[0], x_range[1])
                y = self.np_random.randint(y_range[0], y_range[1])
                barycenters[i] = np.array([x, y])
                if i == 0:
                    loop = False
                else:
                    for k in range(i):
                        if abs(barycenters[i, 0] - barycenters[k, 0]) <= 1 or abs(barycenters[i, 1] - barycenters[k, 1]) <= 1:
                            break
                        else:
                            loop = False

        tmp = barycenters[:n_symbols].copy()
        tmp = sorted(tmp, key=cmp_to_key(partial(self.rule_defined_ordering, rule_name))) # This is where we need to use the rule-based ordering
        barycenters[:n_symbols] = tmp

        # Then, choose the symbols and make the artists
        symbols_int = self.np_random.randint(len(symbol_types), size=(n_symbols))
        artists = [None for _ in range(18)]
        # This will automatically filter the non existent symbols
        for i, barycenter, symbol_int in zip(range(n_symbols), barycenters, symbols_int):
            artists[i] = Artist(barycenter, symbol_types[symbol_int]) 

        # Finally, put the patches on a board
        board = self.backgrounds[rule_name].copy()
        for artist in artists:
            if artist is not None:
                board = artist.draw(board)

        return board, artists, n_symbols, symbols_done
    

    def _reset_one_env(self, env_id):
        # self.rules_idx[env_id] = self.np_random.randint(len(self.rules_names))
        self.rules_idx[env_id] = self.np_random.randint(len(self.allowed_rules))
        self.rules[env_id] = self.allowed_rules[self.rules_idx[env_id]]
        board, artists, n_symbols, symbols_done = self._generate_one_board(rule_idx=self.rules_idx[env_id])
        self.boards[env_id] = board
        tmp = np.array([artist.barycenter if artist is not None else [0., 0.] for artist in artists])
        tmp = tmp / 64. - 1.
        self.barycenters[env_id] = np.array([artist.barycenter / 64. - 1. if artist is not None else [0., 0.] for artist in artists])

        self.target_symbols[env_id] = np.array([artist.draw(np.zeros((self.n_pixels, self.n_pixels, 3))) if artist is not None else np.zeros((self.n_pixels, self.n_pixels, 3)) for artist in artists])
        self.target_endpoints[env_id] = np.array([np.hstack([artist.start / 64. - 1., artist.end / 64. - 1.]) if artist is not None else np.zeros(4) for artist in artists])
        self.artists[env_id] = artists
        self.symbols_done[env_id] = symbols_done
        self.n_symbols[env_id] = n_symbols
        self.timeouts[env_id] = 2*self.n_symbols[env_id]
        self.times[env_id] = 0
        self.positions[env_id] = np.zeros(2)
        self.positions_patch[env_id] = self.default_pos_patch.copy()
        self.boards[env_id] += self.positions_patch[env_id]
        self.epoch_rewards[env_id] = 0.

    def reset(self):
        # NOTE: boards are between 0 and 1, not 0 and 255.
        self.positions = np.zeros((self.n_envs, 2))
        self.timeouts = np.ones(self.n_envs, dtype=int)
        self.symbols_done = np.zeros((self.n_envs, 18), dtype=bool) 
        self.artists = [[None for _ in range(18)] for _ in range(self.n_envs)] # That's how we will get the images
        self.target_symbols = np.zeros((self.n_envs, 18, self.n_pixels, self.n_pixels, 3))
        self.target_endpoints = np.zeros((self.n_envs, 18, 4))
        self.barycenters = np.zeros((self.n_envs, 18, 2), dtype=np.float32)
        self.boards = np.zeros((self.n_envs, self.n_pixels, self.n_pixels, 3), dtype=np.float32)
        self.n_symbols = np.zeros(self.n_envs, dtype=int)
        self.positions_patch = np.stack([self.default_pos_patch for _ in range(self.n_envs)], axis=0)
        self.times = np.zeros(self.n_envs, dtype=int)
        self.epoch_rewards = np.zeros(self.n_envs, dtype=int)
        
        self.rules_idx = np.zeros(self.n_envs, dtype=int)
        self.rules = np.array(['leftward' for _ in range(self.n_envs)], dtype=object)

        for i in range(self.n_envs):
            if i == 0 or not self.all_envs_start_identical: 
                self._reset_one_env(i)
            else:
                self.boards[i] = self.boards[0]
                self.artists[i] = self.artists[0]
                self.barycenters[i] = self.barycenters[0]
                self.target_endpoints[i] = self.target_endpoints[0]
                self.target_symbols[i] = self.target_symbols[0]
                self.symbols_done[i] = self.symbols_done[0]
                self.n_symbols[i] = self.n_symbols[0]
                self.timeouts[i] = self.timeouts[0]
                self.rules_idx[i] = self.rules_idx[0]
                self.rules[i] = self.rules[0]

        self.boards += self.positions_patch
        self.times = np.zeros(self.n_envs, dtype=int)
        self.boards = np.clip(self.boards, 0., 1.)

        self.epoch_rewards = np.zeros((self.n_envs, 2*18))
        return self.boards.copy()
    
    def _draw_on_boards(self, actions):
        # actions is a (bs, 3) array of actions
        velocities, strengths = actions[:, :2], actions[:, 2]

        # Agent positions are continuous, unlike artist positions
        new_positions = self.positions + velocities

        # Clip new positions to be within the board
        new_positions = np.clip(new_positions, -1., 1.)

        # Compute the new positions patch
        new_positions_patch = self._positions_to_patch(new_positions).transpose((0, 2, 1, 3))

        # Compute the new boards
        new_boards = self.boards.copy() - self.positions_patch.copy() + new_positions_patch.copy()
        new_draws = np.zeros((self.n_envs, self.n_pixels, self.n_pixels, 3), dtype=np.uint8)

        # Compute the new lines
        for i in range(self.n_envs):
            if strengths[i] > 0.:
                x1, y1 = (self.n_pixels * (self.positions[i] + 1.) / 2.).astype(int)
                x2, y2 = (self.n_pixels * (new_positions[i] + 1.) / 2.).astype(int)
                new_draws[i] = cv2.line(np.zeros((self.n_pixels, self.n_pixels, 3), dtype=np.uint8), (y1, x1), (y2, x2), (0,0,1), 2)

            new_boards[i] += new_draws[i]
        
        self.boards = np.clip(new_boards, 0., 1.)
        self.positions = new_positions
        self.positions_patch = new_positions_patch

        return new_draws.copy()

    def _compute_reward(self, new_draws):
        rewards = np.zeros(self.n_envs)
        added_pixels = np.any(new_draws[:,:,:,2], axis=(1,2))

        if self.reward_type == 'default':
            added_pixels = (np.sum(new_draws[:,:,:,2], axis=(1,2)) >= 1)
            for board_id in range(self.n_envs):
                if not np.all(self.symbols_done[board_id]):
                    if added_pixels[board_id]:
                        overlaps = []
                        for symbol_id in range(self.n_symbols[board_id]):
                            if not self.symbols_done[board_id, symbol_id]:
                                done_overlap_ref = np.sum(new_draws[board_id, :, :, 2]*self.target_symbols[board_id, symbol_id, :, :, 1]) / (1.+np.sum(new_draws[board_id, :, :, 2]))
                                ref_overlap_done = np.sum(new_draws[board_id, :, :, 2]*self.target_symbols[board_id, symbol_id, :, :, 1]) / (1.+np.sum(self.target_symbols[board_id, symbol_id, :, :, 1]))
                                overlaps.append(min(done_overlap_ref, ref_overlap_done))
                            else:
                                overlaps.append(-.1)

                        if max(overlaps) > self.reward_params.overlap_criterion:
                            # Not done, drew, and completed at least one line
                            # Allow to do several lines at once, should help for case of overlapping lines
                            for symbol_id in range(self.n_symbols[board_id]):
                                if overlaps[symbol_id] > self.reward_params.overlap_criterion:
                                    self.symbols_done[board_id, symbol_id] = True
                                    rewards[board_id] = 1 # Or do we add? I feel like it would encourage bad strategies...
                        else:
                            # Not done, drew, but did not complete a line
                            rewards[board_id] = 0.
                    else:
                        # Not done, but no new pixels added
                        continue
                else:
                    if added_pixels[board_id]:
                        # done, but new pixels added
                        rewards[board_id] = 0.
                    else:
                        # done, and no new pixels added
                        rewards[board_id] = 1.

        self.epoch_rewards[:, self.times-1] = rewards

        return rewards.copy()

    def step(self, actions, done=None):
        assert isinstance(actions, np.ndarray), f'Actions should be a numpy array, not {type(actions)}'
        actions[:, :2] = np.clip(actions[:, :2], -2., 2.)
        actions[:, 2] = np.clip(actions[:, 2], -1., 1.)

        start_times = self.times.copy()

        new_lines = self._draw_on_boards(actions)
        self.times += 1

        rewards = self._compute_reward(new_lines)

        dones = (self.times == self.timeouts)

        terminal_obs = deepcopy(self.boards)
        terminal_pos = [p.copy() for p in self.positions]
       
        for i in range(self.n_envs):
            if dones[i]: 
                self._reset_one_env(i)

        assert np.all(0 <= self.boards) and np.all(self.boards <= 1), f'Boards should be between 0 and 1'

        return self.boards, rewards, dones, dones, [{'terminal_observation': terminal_obs[b], 'terminal_position': terminal_pos[b], 'time': start_times[b]} for b in range(self.n_envs)]



if __name__ == '__main__':
    test_dir = 'out/rule_board_tests/'
    os.makedirs(test_dir, exist_ok=True)

    # Check our artists for the different symbols are working
    for symbol_type in symbol_types:
        for rep in range(3):
            artist_params = {
                'barycenter': np.random.randint(16, 112, size=2), # Make sure it fits on page
                'symbol_type': symbol_type,
            }
            artist = Artist(**artist_params)
            board = np.zeros((128, 128, 3), dtype=int)
            board = artist.draw(board)

            # Transpose comes from the fact that for numpy the rows are y axis and the columns are x axis..... 
            plt.matshow(255*board.transpose(1,0,2), origin='lower')
            plt.scatter(artist.barycenter[0], artist.barycenter[1], c='k', s=40, label='Barycenter')
            plt.scatter(artist.start[0], artist.start[1], c='g', marker='+', s=40, label='Start')
            plt.scatter(artist.end[0], artist.end[1], c='r', marker='+', s=40, label='End')
            plt.legend()
            plt.savefig(test_dir + f'{symbol_type}_{rep}.png')
            plt.close()

    board_params = {
        'n_envs': 12,
        'n_symbols_min': 4,
        'n_symbols_max': 8,
        'reward_type': 'default',
        'reward_params':  {'overlap_criterion': .4},
        'all_envs_start_identical': False,
        'allowed_symbols': ['line_0', 'line_1', 'line_2', 'line_3'],
        'allowed_rules': ['rightward', 'leftward', 'upward', 'downward'],
    }
    boards = RuleBoards(board_params)

    # Check the color borders
    for k,v in boards.rule_border_colors.items():
        plt.figure()
        plt.imshow(boards.backgrounds[k])
        plt.title(f'Rule: {k}')
        plt.savefig(test_dir+f'background_{k}.png')
        plt.close('all')
        

    # Second, make sure the spawn zones are reasonable
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((128, 128, 3), dtype=int), origin='lower')
    for zone_id, spawn_zone in enumerate(boards.spawn_zones):
        ax.text((spawn_zone[0][0] + spawn_zone[0][1])/2, (spawn_zone[1][0] + spawn_zone[1][1])/2, str(zone_id), ha='center', va='center')
        ax.add_patch(patches.Rectangle((spawn_zone[0][0], spawn_zone[1][0]), spawn_zone[0][1]-spawn_zone[0][0], spawn_zone[1][1]-spawn_zone[1][0], linewidth=1, edgecolor='r', facecolor='g'))
    plt.savefig(test_dir + 'spawn_zones.png')


    # Third, make sure the boards are being generated correctly at reset
    boards = RuleBoards(board_params)
    initial_img = boards.reset()
    for env_id in range(10):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(initial_img[env_id].transpose(1,0,2), origin='lower')
        axes[0].set_title('Full observation')
        axes[1].imshow(boards.artists[env_id][0].draw(np.zeros_like(initial_img[env_id])).transpose(1,0,2), origin='lower')
        axes[1].set_title('left-bot most artist (alone)')
        axes[2].imshow(boards.get_centered_patch(env_id, center_idx=boards.artists[env_id][0].barycenter).transpose(1,0,2), origin='lower')
        axes[2].set_title('Punched-in view around barycenter of left-most artist')

        axes[3].imshow(initial_img[env_id].transpose(1,0,2), origin='lower', extent=[-1,1,-1,1])
        for barycenter, end, _ in zip(boards.barycenters[env_id], boards.target_endpoints[env_id], range(boards.n_symbols[env_id])):
            axes[3].plot([end[0], end[2]], [end[1], end[3]], lw=4, label=f"({end[0]:.2f}|{end[1]:.2f}) -> ({end[2]:.2f}|{end[3]:.2f})")
            axes[3].scatter(barycenter[0], barycenter[1], s=100, marker='+', color='gray', label=f"Barycenter: ({barycenter[0]:.2f}|{barycenter[1]:.2f})")
        box = axes[3].get_position()
        axes[3].set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(test_dir + f'reset_imgs_{env_id}.png')
        plt.close(fig)

    board_params['all_envs_start_identical'] = True
    boards = RuleBoards(board_params)
    initial_img = boards.reset()
    for env_id in range(3):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(initial_img[env_id].transpose(1,0,2), origin='lower')
        axes[0].set_title('Full observation')
        axes[1].imshow(boards.artists[env_id][0].draw(np.zeros_like(initial_img[env_id])).transpose(1,0,2), origin='lower')
        axes[1].set_title('left-bot most artist (alone)')
        axes[2].imshow(boards.get_centered_patch(env_id, center_idx=boards.artists[env_id][0].barycenter).transpose(1,0,2), origin='lower')
        axes[2].set_title('Punched-in view around barycenter of left-most artist')
        fig.savefig(test_dir + f'identical_envs_reset_imgs_{env_id}.png')
        plt.close(fig)

    # This one is not very interesting
    # board_params['all_envs_start_identical'] = False
    # board_params['n_symbols_max'] = 18
    # board_params['n_symbols_min'] = 18
    # boards = RuleBoards(board_params)
    # initial_img = boards.reset()
    # for env_id in range(5):
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #     axes[0].imshow(initial_img[env_id].transpose(1,0,2), origin='lower')
    #     axes[0].set_title('Full observation')
    #     axes[1].imshow(boards.get_centered_patch(env_id, center_idx=boards.artists[env_id][np.random.randint(18)].barycenter).transpose(1,0,2), origin='lower')
    #     axes[1].set_title('Punched-in view around barycenter of a random artist')
    #     axes[2].imshow(boards.get_centered_patch(env_id, center_idx=boards.artists[env_id][np.random.randint(18)].barycenter).transpose(1,0,2), origin='lower')
    #     axes[2].set_title('Punched-in view around barycenter of another random artist')
    #     fig.savefig(test_dir + f'dense_reset_imgs_{env_id}.png')
    #     plt.close(fig)

