import numpy as np
from copy import deepcopy
from numpy.random import RandomState
import logging
import torch as tch
import matplotlib
matplotlib.use('Agg')
from matplotlib import patches
import matplotlib.pyplot as plt
import os
from matplotlib import animation
import cv2

# Allows sorting using custon comparator
from functools import cmp_to_key


symbol_types = ['line_0', 'line_1', 'line_2', 'line_3',]
symbols_starts_in_bc_frame = {
    'line_0':  np.array([8, 16]),
    'line_1':  np.array([16, 8]),
    'line_2':  np.array([8, 8]),
    'line_3':  np.array([8, 24]),
}

symbols_ends_in_bc_frame = {
    'line_0':  np.array([24, 16]),
    'line_1':  np.array([16, 24]),
    'line_2':  np.array([24, 24]),
    'line_3':  np.array([24, 8]),
}

# Use always chennel last format for simplicity (h, w, 3)
# Transpose is here for the visual interpretability; doesn't matter for agent as long as positions are kept coherent.
types2patch = {
    'line_0':  cv2.line(np.zeros((32, 32, 3), dtype=np.uint8), (8, 16), (24, 16), (0,0,1), 2).transpose(1, 0, 2),
    'line_1':  cv2.line(np.zeros((32, 32, 3), dtype=np.uint8), (16, 8), (16, 24), (0,0,1), 2).transpose(1, 0, 2),
    'line_2':  cv2.line(np.zeros((32, 32, 3), dtype=np.uint8), (8, 8), (24, 24), (0,0,1), 2).transpose(1, 0, 2),
    'line_3':  cv2.line(np.zeros((32, 32, 3), dtype=np.uint8), (8, 24), (24, 8), (0,0,1), 2).transpose(1, 0, 2),
}


def globalpixel2globalpos(global_pixel):
    return (global_pixel - 64.)/ 64.

def localpixel2localpos(local_pixel):
    return (local_pixel - 16.)/ 16.


class Artist:
    def __init__(self, barycenter=(64, 64), symbol_type='line_0'):
        self.barycenter = barycenter # in pixels, in the global frame
        self.symbol_type = symbol_type
        self.start_in_bc_frame = symbols_starts_in_bc_frame[self.symbol_type]
        self.end_in_bc_frame = symbols_ends_in_bc_frame[self.symbol_type]
        self.start = self.start_in_bc_frame + self.barycenter[0] - 16 # Barycenter is at position 16, 16
        self.end = self.end_in_bc_frame + self.barycenter[1] - 16 # Barycenter is at position 16, 16

    def draw(self, board):
        x, y = self.barycenter
        patch = types2patch[self.symbol_type]
        board[x-16:x+16, y-16:y+16] = (board[x-16:x+16, y-16:y+16] + patch).clip(0,1)
        return board

# This will be for the "global state"   
class Boards:
    def __init__(self, board_params, seed=0):
        self.seed = seed
        self.n_pixels = 128 # Hardcoded as we want a fancy setup.
        self.board_params = board_params 
        self.n_envs = board_params['n_envs']
        self.n_symbols_min = board_params['n_symbols_min']
        self.n_symbols_max = board_params['n_symbols_max']
        self.reward_type = board_params['reward_type']
        self.reward_params = board_params['reward_params']
        self.all_envs_start_identical = board_params['all_envs_start_identical']

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

        self.default_pos_patch = np.zeros((128, 128, 3), dtype=np.uint8)
        self.default_pos_patch = cv2.circle(self.default_pos_patch, (64,64), 2, (1,0,0), -1)

        self.set_seed()
        self.reset()

    def set_seed(self):
        logging.critical(f'Called world.set_seed with seed {self.seed}')
        self.np_random = np.random.RandomState(seed=self.seed)

    def __leq_barycenters(self, b1, b2):
        col1 = b1[0] // 16
        col2 = b2[0] // 16
        row1 = b1[1] // 16
        row2 = b2[1] // 16

        if col1 != col2:
            # return b1[0] >= b2[0]
            return b1[0] - b2[0]
        elif row1 != row2: 
            return b1[1] - b2[1]
        else:
            raise ValueError(f'Both barycenters {b1} and {b2} are in the same spawn zones !!')

    def get_centered_patch(self, env_id, center):
        x, y = center
        return self.boards[env_id, x-16:x+16, y-16:y+16]

    def _positions_to_patch(self, new_positions):
        positions_int = np.array(((new_positions + 1.)/2.*self.n_pixels), dtype=int)
        patches = np.zeros((self.n_envs, 128, 128, 3), dtype=np.uint8)
        for i, pos in enumerate(positions_int):
            patches[i] = cv2.circle(patches[i], tuple(pos), 2, (1,0,0), -1)
        return patches
    
    def _generate_one_board(self):
        n_symbols = self.np_random.randint(self.n_symbols_min, self.n_symbols_max+1)
        symbols_done = np.zeros(18, dtype=bool) # Hardcode max, but might want to use lower
        symbols_done[n_symbols:] = True # Non existent symbols are still considered as symbols, just already done
        barycenters = np.zeros((18, 2), dtype=int)

        # First, draw the spawn zones
        spawn_ranges = self.np_random.permutation(self.spawn_zones)[:n_symbols]

        # print('spawn_ranges', spawn_ranges)
        for i, (x_range, y_range) in enumerate(spawn_ranges):
            x = self.np_random.randint(x_range[0], x_range[1])
            y = self.np_random.randint(y_range[0], y_range[1])
            barycenters[i] = np.array([x, y])

        tmp = barycenters[:n_symbols].copy()
        tmp = sorted(tmp, key=cmp_to_key(self.__leq_barycenters))
        barycenters[:n_symbols] = tmp

        # Then, choose the symbols and make the artists
        symbols_int = self.np_random.randint(len(symbol_types), size=(n_symbols))
        artists = [None for _ in range(18)]
        # This will automatically filter the non existent symbols
        for i, barycenter, symbol_int in zip(range(n_symbols), barycenters, symbols_int):
            artists[i] = Artist(barycenter, symbol_types[symbol_int]) 

        # print('in generate one board', artists)

        # Finally, put the patches on a board
        board = np.zeros((128, 128, 3), dtype=np.uint8)     
        for artist in artists:
            if artist is not None:
                board = artist.draw(board)

        return board, artists, n_symbols, symbols_done
    

    def _reset_one_env(self, env_id):
        board, artists, n_symbols, symbols_done = self._generate_one_board()
        self.boards[env_id] = board
        self.barycenters[env_id] = [artist.barycenter for artist in artists]
        self.artists[env_id] = artists
        self.symbols_done[env_id] = symbols_done
        self.n_symbols[env_id] = n_symbols
        self.timeouts[env_id] = self.n_symbols[env_id]
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
        self.barycenters = [[None for _ in range(18)] for _ in range(self.n_envs)]
        self.boards = np.zeros((self.n_envs, self.n_pixels, self.n_pixels, 3), dtype=np.uint8)
        self.n_symbols = np.zeros(self.n_envs, dtype=int)
        self.positions_patch = np.stack([self.default_pos_patch for _ in range(self.n_envs)], axis=0)
        # Everyone starts at center of drawing board no matter what.

        for i in range(self.n_envs):
            if i == 0 or not self.all_envs_start_identical: 
                board, artists, n_symbols, symbols_done = self._generate_one_board()
                self.boards[i] = board
                self.artists[i] = artists
                self.barycenters[i] = np.array([artist.barycenter if artist is not None else (0., 0.) for artist in artists])
                self.symbols_done[i] = symbols_done
                self.n_symbols[i] = n_symbols
                self.timeouts[i] = 2*n_symbols 
            else:
                self.boards[i] = self.boards[0]
                self.artists[i] = self.artists[0]
                self.barycenters[i] = self.barycenters[0]
                self.symbols_done[i] = self.symbols_done[0]
                self.n_symbols[i] = self.n_symbols[0]
                self.timeouts[i] = self.timeouts[0]

        self.boards += self.positions_patch
        self.times = np.zeros(self.n_envs, dtype=int)
        self.boards = np.clip(self.boards, 0., 1.)

        self.epoch_rewards = np.zeros((self.n_envs, 2*18))
        return self.boards.copy()
    
#     def _draw_on_boards(self, actions):
#         # actions is a (bs, 3) array of actions
#         velocities, strengths = actions[:, :2], actions[:, 2]
#         new_positions = self.positions + velocities

#         # Clip new positions to be within the board
#         new_positions = np.clip(new_positions, -1., 1.)

#         # Compute the new positions patch
#         new_positions_patch = self._positions_to_patch(new_positions)

#         # Compute the new boards
#         new_boards = self.boards - self.positions_patch + new_positions_patch
#         new_lines = np.zeros((self.n_envs, self.n_pixels, self.n_pixels, 3), dtype=np.uint8)

#         # Compute the new lines
#         for i in range(self.n_envs):
#             new_line = np.zeros((self.n_pixels, self.n_pixels, 3), dtype=np.uint8)
#             if strengths[i] > 0.:

#                 if strengths[i] > .5:
#                     w = 5
#                 elif strengths[i] > .25:
#                     w = 3
#                 else:
#                     w = 1

#                 x1, y1 = (self.n_pixels * (self.positions[i] + 1.) / 2.).astype(int)
#                 x2, y2 = (self.n_pixels * (new_positions[i] + 1.) / 2.).astype(int)
#                 new_line = cv2.line(new_line, (x1, y1), (x2, y2), (0,0,1), w)
#                 new_lines[i] = deepcopy(new_line)
#             new_boards[i] += new_line

#         new_boards = np.clip(new_boards, 0., 1.)
        
#         self.boards = new_boards
#         self.positions = new_positions
#         self.positions_patch = new_positions_patch

#         return new_lines

#     def _compute_reward(self, new_lines):
#         rewards = np.zeros(self.n_envs)
#         added_pixels = np.any(new_lines[:,:,:,2], axis=(1,2))

#         if self.reward_type == 'default':
#             added_pixels = (np.sum(new_lines[:,:,:,2], axis=(1,2)) >= 1)
#             for board_id in range(self.n_envs):
#                 if not np.all(self.lines_done[board_id]):
#                     if added_pixels[board_id]:
#                         overlaps = []
#                         for line_id in range(self.n_lines[board_id]):
#                             if not self.lines_done[board_id, line_id]:
#                                 line_overlap_ref = np.sum(new_lines[board_id, :, :, 2]*self.target_lines[board_id, line_id, :, :, 1]) / (1.+np.sum(new_lines[board_id, :, :, 2]))
#                                 ref_overlap_line = np.sum(new_lines[board_id, :, :, 2]*self.target_lines[board_id, line_id, :, :, 1]) / (1.+np.sum(self.target_lines[board_id, line_id, :, :, 1]))
#                                 overlaps.append(min(line_overlap_ref, ref_overlap_line))
#                             else:
#                                 overlaps.append(-.1)

#                         if max(overlaps) > self.overlap_criterion:
#                             # Not done, drew, and completed at least one line
#                             # Allow to do several lines at once, should help for case of overlapping lines
#                             for line_id in range(self.n_lines[board_id]):
#                                 if overlaps[line_id] > self.overlap_criterion:
#                                     self.lines_done[board_id, line_id] = True
#                                     rewards[board_id] = 1 # Or do we add? I feel like it would encourage bad strategies...
#                         else:
#                             # Not done, drew, but did not complete a line
#                             rewards[board_id] = 0.
#                     else:
#                         # Not done, but no new pixels added
#                         continue
#                 else:
#                     if added_pixels[board_id]:
#                         # done, but new pixels added
#                         rewards[board_id] = 0.
#                     else:
#                         # done, and no new pixels added
#                         rewards[board_id] = 1.

#         self.epoch_rewards[:, self.times-1] = rewards

#         return rewards.copy()

#     def step(self, actions, done=None):
#         assert isinstance(actions, np.ndarray), f'Actions should be a numpy array, not {type(actions)}'
#         actions[:, :2] = np.clip(actions[:, :2], -2., 2.)
#         actions[:, 2] = np.clip(actions[:, 2], -1., 1.)

#         start_times = self.times.copy()

#         new_lines = self._draw_on_boards(actions)
#         self.times += 1

#         rewards = self._compute_reward(new_lines)

#         dones = (self.times == self.timeouts)

#         terminal_obs = deepcopy(self.boards)
#         terminal_pos = [p.copy() for p in self.positions]
       
#         for i in range(self.n_envs):
#             if dones[i]: 
#                 self._reset_one_env(i)

#         assert np.all(0 <= self.boards) and np.all(self.boards <= 1), f'Boards should be between 0 and 1'

#         return self.boards, rewards, dones, dones, [{'terminal_observation': terminal_obs[b], 'terminal_position': terminal_pos[b], 'time': start_times[b]} for b in range(self.n_envs)]

if __name__ == '__main__':
    test_dir = 'out/board_tests/'
    os.makedirs(test_dir, exist_ok=True)


    # First, make sure our artists for the different symbols are working
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
        'reward_params': {},
        'all_envs_start_identical': False,
    }

    # Second, make sure the spawn zones are reasonable
    boards = Boards(board_params)
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((128, 128, 3), dtype=int), origin='lower')
    for zone_id, spawn_zone in enumerate(boards.spawn_zones):
        ax.text((spawn_zone[0][0] + spawn_zone[0][1])/2, (spawn_zone[1][0] + spawn_zone[1][1])/2, str(zone_id), ha='center', va='center')
        ax.add_patch(patches.Rectangle((spawn_zone[0][0], spawn_zone[1][0]), spawn_zone[0][1]-spawn_zone[0][0], spawn_zone[1][1]-spawn_zone[1][0], linewidth=1, edgecolor='r', facecolor='g'))
    plt.savefig(test_dir + 'spawn_zones.png')


    # Third, make sure the boards are being generated correctly at reset
    boards = Boards(board_params)
    initial_img = boards.reset()
    for env_id in range(10):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(initial_img[env_id].transpose(1,0,2), origin='lower')
        axes[0].set_title('Full observation')
        axes[1].imshow(boards.artists[env_id][0].draw(np.zeros_like(initial_img[env_id])).transpose(1,0,2), origin='lower')
        axes[1].set_title('left-bot most artist (alone)')
        axes[2].imshow(boards.get_centered_patch(env_id, boards.artists[env_id][0].barycenter).transpose(1,0,2), origin='lower')
        axes[2].set_title('Punched-in view around barycenter of left-most artist')
        fig.savefig(test_dir + f'reset_imgs_{env_id}.png')
        plt.close(fig)

    board_params['all_envs_start_identical'] = True
    boards = Boards(board_params)
    initial_img = boards.reset()
    for env_id in range(10):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(initial_img[env_id].transpose(1,0,2), origin='lower')
        axes[0].set_title('Full observation')
        axes[1].imshow(boards.artists[env_id][0].draw(np.zeros_like(initial_img[env_id])).transpose(1,0,2), origin='lower')
        axes[1].set_title('left-bot most artist (alone)')
        axes[2].imshow(boards.get_centered_patch(env_id, boards.artists[env_id][0].barycenter).transpose(1,0,2), origin='lower')
        axes[2].set_title('Punched-in view around barycenter of left-most artist')
        fig.savefig(test_dir + f'identical_envs_reset_imgs_{env_id}.png')
        plt.close(fig)


    board_params['all_envs_start_identical'] = False
    board_params['n_symbols_max'] = 18
    board_params['n_symbols_min'] = 18
    boards = Boards(board_params)
    initial_img = boards.reset()
    for env_id in range(5):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(initial_img[env_id].transpose(1,0,2), origin='lower')
        axes[0].set_title('Full observation')
        axes[1].imshow(boards.get_centered_patch(env_id, boards.artists[env_id][np.random.randint(18)].barycenter).transpose(1,0,2), origin='lower')
        axes[1].set_title('Punched-in view around barycenter of a random artist')
        axes[2].imshow(boards.get_centered_patch(env_id, boards.artists[env_id][np.random.randint(18)].barycenter).transpose(1,0,2), origin='lower')
        axes[2].set_title('Punched-in view around barycenter of another random artist')
        fig.savefig(test_dir + f'dense_reset_imgs_{env_id}.png')
        plt.close(fig)






                        






# class Oracle:
#     # Works on batched environments
#     def __init__(self, sensitivity=.1):
#         self.sensitivity = sensitivity

#     def _fuzzy_2d_points_cmp(self, a, b):
#         # Make sure that if first coordinates are close, we compare the second; 
#         # This should make the edge cases much easier 

#         # Returns a negative number if a is leftmost (as expected by python sort keys)
#         assert a.shape == b.shape == (2,)
#         if abs(a[0] - b[0]) <= self.sensitivity:
#             return a[1] - b[1]
#         else:
#             return a[0] - b[0]

#     @staticmethod
#     def argsort(seq):
#         # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
#         return sorted(range(len(seq)), key=seq.__getitem__)

#     def get_action(self, env):
#         n_envs = env.n_envs
#         actions = np.zeros((n_envs, 3))
#         ref_positions = deepcopy(env.positions)
#         logging.debug(env.positions)
#         for env_idx in range(n_envs):
#             lines_endpoints = env.lines_endpoints[env_idx]
#             lines_done = env.lines_done[env_idx]

#             if np.all(lines_done):
#                 # If all lines are done, no need to continue drawing / moving
#                 actions[env_idx] = [0, 0, -0.1]
#                 continue
#             else:
#                 # Find the closest line endpoint
#                 x, y = ref_positions[env_idx]
#                 best_line_idx = None

#                 end_dists = []
#                 start_dists = []


#                 for line_idx, (x1, y1, x2, y2) in enumerate(lines_endpoints):
#                     if env.lines_done[env_idx, line_idx]:
#                         # Make sure we don't consider the (0,0) endpoints of absent lines here...
#                         start_dists.append(2**10)
#                         end_dists.append(2**10)
#                     else:
#                         start_dists.append(np.sqrt((x1-x)**2 + (y1-y)**2))
#                         end_dists.append(np.sqrt((x2-x)**2 + (y2-y)**2))

#                         # TODO: Make this work with the rule "if there's a clear closest line, take it, otherwise of all those that are within the sensitivity, take the "fuzzily leftmost one"
#                         # if min(start_dist, end_dist) < best_goal_dist:
#                         #     # If it's a fair win, take the closest
#                         #     if min(start_dist, end_dist) < best_goal_dist - self.sensitivity or self._fuzzy_2d_points_leq([x1, y1], [x2, y2]):
#                         #         best_goal_dist = min(start_dist, end_dist)
#                         #         best_goal_location = [x1, y1] if start_dist < end_dist else [x2, y2]
#                         #         best_line_idx = line_idx

#                 best_dist = min(min(start_dists), min(end_dists))
#                 valid_candidates = []
#                 line_ids = [] 

#                 for i, start_dist, end_dist in zip(range(5), start_dists, end_dists):
#                     if start_dist < best_dist + self.sensitivity:
#                         valid_candidates.append(lines_endpoints[i, :2])
#                         line_ids.append(i)
#                     if end_dist < best_dist + self.sensitivity:
#                         valid_candidates.append(lines_endpoints[i, 2:])
#                         line_ids.append(i)

#                 valid_candidates_as_keys = [cmp_to_key(self._fuzzy_2d_points_cmp)(c) for c in valid_candidates]
#                 leftmost_candidate_idx = self.argsort(valid_candidates_as_keys)[0]
#                 best_line_idx = line_ids[leftmost_candidate_idx]
#                 best_goal_location = valid_candidates[leftmost_candidate_idx]

#                 # Now we have the best goal location, we need to compute the action
#                 dx = best_goal_location[0] - x
#                 dy = best_goal_location[1] - y

#                 # Two possibilities: 
#                 # 1. We are already on the line, in which case we just need to move along it
#                 # 2. We are not on the line, in which case we need to move to the nearest endpoint without drawing
#                 # print(np.abs([dx, dy]))
#                 if np.max(np.abs([dx, dy])) <= self.sensitivity:
#                     if np.all(best_goal_location == lines_endpoints[best_line_idx, :2]):
#                         # We are at the start of the line, goal is the end
#                         actual_goal = lines_endpoints[best_line_idx, 2:]
#                     elif np.all(best_goal_location == lines_endpoints[best_line_idx, 2:]):
#                         # We are at the end of the line, goal is the start
#                         actual_goal = lines_endpoints[best_line_idx, :2]
#                     else:
#                         raise RuntimeError('This should not happen')

#                     dx = actual_goal[0] - x
#                     dy = actual_goal[1] - y
#                     best_goal_location = actual_goal
#                     actions[env_idx, 2] = .3
#                 else:
#                     actions[env_idx, 2] = -0.1

#                 actions[env_idx, 0] = dx
#                 actions[env_idx, 1] = dy



            
#         logging.debug(msg=f'Actions at output of oracle: {actions[0]}')
                

#         return actions
 





# if __name__ == '__main__':
#     os.makedirs('out/board_diagnostics', exist_ok=True)

#     board_params = {'n_pixels': 64, 'min_n_lines': 1, 'max_n_lines': 5, 'min_line_length': .5, 'max_line_length': 1., 'reward_type': 'stroke_done_no_penalties', 'overlap_criterion': .3, 'ordering_sensitivity': .07, 'n_envs': 10}
#     board_params['reward_type'] = 'disappearing_lines'

#     env = Boards(board_params, 777)
#     v_noise = .05 
#     v_noise_clip = .1
#     s_noise = .07
#     oracle = Oracle(sensitivity=v_noise_clip)

#     # Now, do a test of oracle in an open loop setting
#     t_tot = 200


#     plot_observations_0 = [] 
#     plot_observations_1 = [] 
#     rewards_0 = []
#     rewards_1 = []

#     next_obs = env.reset()
#     next_done = np.zeros(env.n_envs)


#     for step in range(0, t_tot):
#         plot_observations_0.append(next_obs[0])
#         plot_observations_1.append(next_obs[1])
#         action = oracle.get_action(env)
#         action[1, :2] += np.random.normal(0, s_noise, size=(2)).clip(-v_noise_clip, v_noise_clip)
#         action[1, 2] += np.random.normal(0, v_noise)

#         next_obs, reward, next_done, truncated, info = env.step(action, next_done)
#         rewards_0.append(reward[0])
#         rewards_1.append(reward[1])

#         if next_done[0]:
#             plot_observations_0.append(info[0]['terminal_observation'])
#             rewards_0.append('nan')
#         if next_done[1]:
#             plot_observations_1.append(info[1]['terminal_observation'])
#             rewards_1.append('nan')


#     # Check if we have the "repeating board" bug here
#     savepath='out/board_diagnostics/open_loop'
#     os.makedirs(savepath+ '_unperturbed', exist_ok=True)

#     for t in range(len(plot_observations_0)):
#         fig, ax = plt.subplots()
#         ax.imshow(plot_observations_0[t], origin='lower', extent=[-1, 1, -1, 1])
#         if rewards_0[t] != 'nan':
#             ax.set_title(f'Unperturbed, t {t}, upcoming reward {rewards_0[t]}')
#         else:
#             ax.set_title(f'Unperturbed, t {t}, terminal observation')

#         plt.savefig(savepath + '_unperturbed' + f'/gif_{t}.png')
#         plt.close('all')

#     os.makedirs(savepath + f'_noises_{v_noise:.2e}_{s_noise:.2e}', exist_ok=True)
#     for t in range(len(plot_observations_1)):
#         fig, ax = plt.subplots()
#         ax.imshow(plot_observations_1[t], origin='lower', extent=[-1, 1, -1, 1])
#         if rewards_1[t] != 'nan':
#             ax.set_title(f'With noises {(v_noise, s_noise)}, t {t}, upcoming reward {rewards_1[t]}')
#         else:
#             ax.set_title(f'With noises {(v_noise, s_noise)}, t {t}, terminal observation')

#         plt.savefig(savepath + f'_noises_{v_noise:.2e}_{s_noise:.2e}' + f'/gif_{t}.png')
#         plt.close('all')