import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import seaborn as sns
from scipy.stats import ks_2samp
from itertools import product
import os

# Local vs cluster Shnanigans

# wdir = '/scratch/atf6569/saccade_drawing/'
wdir = '/home/arnaud/Scratch/saccade_drawing/'


OUT_DIR = wdir + 'all_rules/'   


def do_comparisons():
    p = sns.color_palette('muted')
    cdict = {
        'blue': p[0],
        'orange': p[1],
        'green': p[2],
        'red': p[3],
        'purple': p[4],
        'brown': p[5],
        'pink': p[6],
        }
    p_pastel = sns.color_palette('pastel')

    cdict_pastel = {
        'blue': p_pastel[0],
        'orange': p_pastel[1],
        'green': p_pastel[2],
        'red': p_pastel[3],
        'purple': p_pastel[4],
        'brown': p_pastel[5],
        'pink': p_pastel[6],
        }

    # def load_all_seeds_results(template_path, n_seeds=4, out_shape=None):
    #     # template_path should be something like 'results/oracle__cond__only_four/{}/reciprocal_overlaps.npy'
    #     all_seeds_results = []
    #     for seed in range(n_seeds):
    #         all_seeds_results.append(np.load(template_path.format(seed), allow_pickle=True))
    #     if out_shape is None:
    #         all_seeds_results = np.array(all_seeds_results).flatten()
    #     else:
    #         all_seeds_results = np.array(all_seeds_results).reshape(out_shape)
    #     return all_seeds_results

    def load_all_seeds_results(template_path, n_seeds=4, out_shape=None):
        # template_path should be something like 'results/oracle__cond__only_four/{}/reciprocal_overlaps.npy'
        all_seeds_results = []
        for seed in range(n_seeds):
            all_seeds_results.append(np.load(template_path.format(seed), allow_pickle=True))

        all_seeds_results = np.array(all_seeds_results)

        print(all_seeds_results.shape)

        if out_shape is None:
            return all_seeds_results.flatten()
        elif all_seeds_results.shape != out_shape:
            return np.array(all_seeds_results).reshape(out_shape)
        else:
            return all_seeds_results


    def compare_to_baselines():
        # Use the oracle as our reference for "perfection"
        # No real need for "random" reference, it's a delta at 0, use it only in the very first plot


        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        random = load_all_seeds_results(OUT_DIR + 'results/random__cond__only_four/{}/reciprocal_overlaps.npy')
        oracle = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__only_four/{}/reciprocal_overlaps.npy')

        ax = axes[0]
        ax.set_title('Candidate vs baselines (In Distribution, 4 lines)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1]
        random = load_all_seeds_results(OUT_DIR + 'results/random__cond__only_three/{}/reciprocal_overlaps.npy')
        oracle = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__only_three/{}/reciprocal_overlaps.npy')
        candidate = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__only_three/{}/reciprocal_overlaps.npy')
        ax.set_title('Candidate vs baselines (Interpolation, 3 lines)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[2]
        random = load_all_seeds_results(OUT_DIR + 'results/random__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        oracle = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        candidate = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        ax.set_title('Candidate vs baselines (Extrapolation, 5/6 mirrored)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/baselines_comparisons.pdf', dpi=300)

    

    def compare_reward_envelopes():
        # Do it on 8 lines, very long timeout 
        # For "all_rules", we might want to split rules, check if "closest" has some specificities
        # eg it always completes all lines if you leave it enough time ?
        random = load_all_seeds_results(OUT_DIR + 'results/random__cond__mirrored_6_timeout_30/{}/cumulated_rewards.npy')
        random_times = load_all_seeds_results(OUT_DIR + 'results/random__cond__mirrored_6_timeout_30/{}/times.npy')

        oracle = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__mirrored_6_timeout_30/{}/cumulated_rewards.npy')
        oracle_times = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__mirrored_6_timeout_30/{}/times.npy')

        candidate = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__mirrored_6_timeout_30/{}/cumulated_rewards.npy')
        candidate_times = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__mirrored_6_timeout_30/{}/times.npy')

        def do_heatmaps(name, data, times, ax):
            # Two different colors needed to make the filled area readable
            # Easy to get from the seaborn colormaps
            unique_times = np.unique(times) 
            assert np.all(unique_times == np.arange(0, 30))
            means, stds = [], []

            all_counts = np.zeros((30, int(max(data))+1)) # 30 steps, 9 possibilities for cumulated reward
            

            for t in unique_times:
                filtered_data = data[times == t]
                counts = np.bincount(filtered_data.astype(int), minlength=int(max(data+1)))
                all_counts[t] = counts
                means.append(np.mean(filtered_data))
                stds.append(np.std(filtered_data))
            
            all_counts = all_counts.astype(float)
            probs = all_counts / np.sum(all_counts, axis=1, keepdims=True)
            
            sns.heatmap(probs.T[::-1, :], yticklabels=range(int(max(data)), -1, -1), linewidths=.5, annot=True, fmt='.2f', ax=ax)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Number of lines completed')
            ax.set_title(f"Using {name} agent")

        def do_avg_completion(name, data, times, ax, color_strong, color_weak):
            # Two different colors needed to make the filled area readable
            # Easy to get from the seaborn colormaps
            unique_times = np.unique(times) 
            assert np.all(unique_times == np.arange(0, 30))
            means, stds = [], []

            all_counts = np.zeros((30, int(max(data))+1)) # 30 steps, 9 possibilities for cumulated reward
            
            for t in unique_times:
                filtered_data = data[times == t]
                counts = np.bincount(filtered_data.astype(int), minlength=int(max(data)+1))
                all_counts[t] = counts
                means.append(np.mean(filtered_data))
                stds.append(np.std(filtered_data))

            ax.plot(unique_times, means, color=color_strong, label=name)
            ax.fill_between(unique_times, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color_weak, alpha=.3)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Mean number of lines completed')

        fig, axes = plt.subplots(3, 1, figsize=(16, 20))
        all_names = ['Random', 'Oracle', 'Candidate']
        all_data = [random, oracle, candidate]
        all_times = [random_times, oracle_times, candidate_times]
        all_colors = ['brown', 'blue', 'green']
        for name, data, times, ax, cname in zip(all_names, all_data, all_times, axes, all_colors):
            do_heatmaps(name, data, times, ax)
        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/completed_lines_envelope_comparisons.pdf')


        fig, summary_ax = plt.subplots(1, 1, figsize=(8, 6))
        for name, data, times, cname in zip(all_names, all_data, all_times, all_colors):
            # Left panel will look like garbage, but right one should be ok
            do_avg_completion(name, data, times, summary_ax, cdict[cname], cdict_pastel[cname])
        summary_ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/average_completed_lines_comparisons_to_baselines.pdf')

    # Do properties of the trajectories change significantly between rules ?
    # Two plots: 
    # - average number of lines completed over time (one color per rule) on 8 rules long timeout
    # - Saccade positions (superimposed kdes? Align only end-point, only start-point, or both ?). Do on 4 only
    def compare_completion_rate_across_rules():
        rewards = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__mirrored_6_timeout_30/{}/cumulated_rewards.npy')
        times = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__mirrored_6_timeout_30/{}/times.npy')
        rules = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__mirrored_6_timeout_30/{}/rules.npy')

        def do_avg_completion(name, data, times, ax, color_strong, color_weak):
            # Two different colors needed to make the filled area readable
            # Easy to get from the seaborn colormaps
            unique_times = np.unique(times) 
            assert np.all(unique_times == np.arange(0, 30))
            assert max(data) <= 8 # 8 lines, so should get at most 8 rewards in a trajectory
            means, stds = [], []

            all_counts = np.zeros((30, 9)) # 30 steps, 9 possibilities for cumulated reward
            
            for t in unique_times:
                filtered_data = data[times == t]
                counts = np.bincount(filtered_data.astype(int), minlength=9)
                all_counts[t] = counts
                means.append(np.mean(filtered_data))
                stds.append(np.std(filtered_data))

            ax.plot(unique_times, means, color=color_strong, label=name)
            ax.fill_between(unique_times, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color_weak, alpha=.3)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Mean number of lines completed')

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for rule, cname in zip(['closest', 'rightward', 'leftward', 'upward', 'downward'], ['blue', 'orange', 'green', 'pink', 'purple']):
            do_avg_completion(rule, rewards[rules == rule], times[rules == rule], ax, cdict[cname], cdict_pastel[cname])

        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/average_completed_lines_comparisons_across_rules.pdf')

    def compare_saccades_across_rules():
        condition = 'only_four'

        rewards = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/cumulated_rewards.npy')
        times = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/times.npy')
        rules = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/rules.npy')
        saccades = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_saccades.npy', out_shape=(-1, 3))
        homings = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_homings.npy', out_shape=(-1, 3))
        actions = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_actions.npy', out_shape=(-1, 3))
        recomputed_actions = saccades + homings
        # print(np.mean(np.abs(recomputed_actions - actions)), np.max(np.abs(recomputed_actions - actions)), np.min(np.abs(recomputed_actions - actions)))
        oracle_saccades = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_oracle_saccades.npy', out_shape=(-1, 2))
        oracle_actions = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_oracle_actions.npy', out_shape=(-1, 3))

        recomputed_action_errors = np.sqrt(np.mean((recomputed_actions - oracle_actions)**2, axis=-1))
        # print('Recomputed action errors', recomputed_action_errors.shape, recomputed_action_errors.mean(), recomputed_action_errors.std())


        action_errors = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/errors_homing.npy')
        # print('Action errors', action_errors.shape, action_errors.mean(), action_errors.std())
        
        saccade_errors = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/errors_saccade.npy')
        print('Saccades shape', saccades.shape)
        print('Oracle saccades shape', oracle_saccades.shape)
        print('Actions shape', actions.shape)
        print('Rules shape', rules.shape)

        # Many different plots possible here, we'll see what wins in the end

        # First try: in the frame of the desired action 

        # Split between movements where the agent draws, and ones where it does not
        is_drawing = (actions[:, 2] >= 0)
        is_not_drawing = (actions[:, 2] < 0)

        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        for rule_idx, rule, cname in zip(range(5), ['closest', 'rightward', 'leftward', 'upward', 'downward'], ['blue', 'orange', 'green', 'pink', 'purple']):
            rule_selector = (rules == rule)
            rule_and_draw = rule_selector & is_drawing
            rule_and_not_draw = rule_selector & is_not_drawing

            data = saccades[rule_and_draw, :2]  # Saccades in a frame centered around the end of the line (up to some noise)

            if rule != 'closest':
                # sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[0, rule_idx], color=cdict[cname], fill=True)
                axes[0, rule_idx].scatter(x=data[:, 0], y=data[:, 1], color=cdict[cname], s=4, alpha=0.1, rasterized=True)
                axes[0, rule_idx].legend(handles=[plt.Line2D([0], [0], color=cdict[cname], lw=4, label=rule.title())], loc='upper left')
                axes[0, rule_idx].set_title(f'{rule.title()} rule')
            else:
                # Need to subdivide the closest rule into 2, depending on whether we draw the line in the "correct" direction or not
                # Note that this is only for drawing steps.
                significantly_leftward = saccades[rule_and_draw, 0] < -0.02
                significantly_rightward = saccades[rule_and_draw, 0] > 0.02
                significantly_upward = saccades[rule_and_draw, 1] > 0.02
                significantly_downward = saccades[rule_and_draw, 1] < -0.02

                usual_direction = significantly_rightward | significantly_upward  
                other_direction = significantly_leftward | significantly_downward
                print('Usual direction', np.sum(usual_direction), 'Other direction', np.sum(other_direction), 'Both', np.sum(usual_direction & other_direction), 'None', np.sum(~usual_direction & ~other_direction))

                usual_direction_data = data[usual_direction & ~other_direction]
                other_direction_data = data[other_direction & ~usual_direction]

                axes[0, rule_idx].scatter(x=usual_direction_data[:, 0], y=usual_direction_data[:, 1], color=cdict['blue'], s=4, alpha=0.1, rasterized=True)
                axes[0, rule_idx].scatter(x=other_direction_data[:, 0], y=other_direction_data[:, 1], color=cdict['red'], s=4, alpha=0.1, rasterized=True)

                axes[0, rule_idx].legend(handles=[plt.Line2D([0], [0], color=cdict['blue'], lw=4, label='Usual (left to right)'), plt.Line2D([0], [0], color=cdict['red'], lw=4, label='Unusual (right to left)')], loc='upper left')

                frac_useless = 1. - (len(usual_direction_data) + len(other_direction_data))/len(data)
                axes[0, rule_idx].set_title(f'Closest rule\n ({np.round(frac_useless, 2)*100}% of steps have no clear direction)')
                
            axes[0, rule_idx].set_xlabel('X saccade (drawing steps)')
            axes[0, rule_idx].set_ylabel('Y saccade (drawing steps)')

            # data = saccades[rule_and_not_draw, :2] - oracle_actions[rule_and_not_draw, :2] # Saccades in a frame centered around the end of the line (up to some noise)
            data = saccades[rule_and_not_draw, :2] - oracle_saccades[rule_and_not_draw, :2] 
            endpos = actions[rule_and_not_draw, :2] - oracle_saccades[rule_and_not_draw, :2]
            # sns.kdeplot(x=data[:, 0], y=data[:, 1], color=cdict[cname], fill=True, ax=axes[1, rule_idx])
            axes[1, rule_idx].scatter(x=data[:, 0], y=data[:, 1], color=cdict[cname], s=1, alpha=0.1, rasterized=True)
            axes[1, rule_idx].legend(handles=[plt.Line2D([0], [0], color=cdict[cname], lw=4, label=rule.title())], loc='upper left')


            axes[1, rule_idx].set_xlabel('X saccade (movement steps, centered on next line)')
            axes[1, rule_idx].set_ylabel('Y saccade (movement steps, centered on next line)')

            # axes[0, rule_idx].set_title('Steps where agent draws')
            axes[0, rule_idx].set_xlim(-.25, .25)
            axes[0, rule_idx].set_ylim(-.25, .25)
            axes[0, rule_idx].axvline(0, color='black', linestyle='--', lw=1)
            axes[0, rule_idx].axhline(0, color='black', linestyle='--', lw=1)
            axes[0, rule_idx].set_aspect('equal')
            axes[1, rule_idx].set_title('Steps where agent does not draw')
            axes[1, rule_idx].set_xlim(-.5, .5)
            axes[1, rule_idx].set_ylim(-.5, .5)
            axes[1, rule_idx].axvline(0, color='black', linestyle='--', lw=1)
            axes[1, rule_idx].axhline(0, color='black', linestyle='--', lw=1)
            axes[1, rule_idx].set_aspect('equal')
        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/scatter_saccades_comparisons_across_rules_frame_oracle_action.pdf')

        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        for rule_idx, rule, cname in zip(range(5), ['closest', 'rightward', 'leftward', 'upward', 'downward'], ['blue', 'orange', 'green', 'pink', 'purple']):
            rule_selector = (rules == rule)
            # Need downsampling, otherwise kde takes forever...
            rule_selector = rule_selector & (np.random.rand(len(rule_selector)) < 0.1)
            rule_and_draw = rule_selector & is_drawing
            rule_and_not_draw = rule_selector & is_not_drawing

            data = saccades[rule_and_draw, :2]  # Saccades in a frame centered around the end of the line (up to some noise)


            if rule != 'closest':
                sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[0, rule_idx], color=cdict[cname], fill=True)
                axes[0, rule_idx].legend(handles=[plt.Line2D([0], [0], color=cdict[cname], lw=4, label=rule.title())], loc='upper left')
                axes[0, rule_idx].set_title(f'{rule.title()} rule')
            else:
                # Need to subdivide the closest rule into 2, depending on whether we draw the line in the "correct" direction or not
                # Note that this is only for drawing steps.
                significantly_leftward = saccades[rule_and_draw, 0] < -0.02
                significantly_rightward = saccades[rule_and_draw, 0] > 0.02
                significantly_upward = saccades[rule_and_draw, 1] > 0.02
                significantly_downward = saccades[rule_and_draw, 1] < -0.02

                usual_direction = significantly_rightward | significantly_upward  
                other_direction = significantly_leftward | significantly_downward
                print('Usual direction', np.sum(usual_direction), 'Other direction', np.sum(other_direction), 'Both', np.sum(usual_direction & other_direction), 'None', np.sum(~usual_direction & ~other_direction))

                usual_direction_data = data[usual_direction & ~other_direction]
                other_direction_data = data[other_direction & ~usual_direction]
                sns.kdeplot(x=usual_direction_data[:, 0], y=usual_direction_data[:, 1], ax=axes[0, rule_idx], color=cdict['blue'], fill=True)
                sns.kdeplot(x=other_direction_data[:, 0], y=other_direction_data[:, 1], ax=axes[0, rule_idx], color=cdict['red'], fill=True)

                frac_useless = 1. - (len(usual_direction_data) + len(other_direction_data))/len(data)
                axes[0, rule_idx].set_title(f'Closest rule\n ({np.round(frac_useless, 2)*100}% of steps have no clear direction)')
                
            axes[0, rule_idx].set_xlabel('X saccade (drawing steps)')
            axes[0, rule_idx].set_ylabel('Y saccade (drawing steps)')

            print('starting second plot')

            # data = saccades[rule_and_not_draw, :2] - oracle_actions[rule_and_not_draw, :2] # Saccades in a frame centered around the end of the line (up to some noise)
            data = saccades[rule_and_not_draw, :2] - oracle_saccades[rule_and_not_draw, :2] 
            # endpos = actions[rule_and_not_draw, :2] - oracle_saccades[rule_and_not_draw, :2]
            sns.kdeplot(x=data[:, 0], y=data[:, 1], color=cdict[cname], fill=True, ax=axes[1, rule_idx])
            # axes[1, rule_idx].scatter(x=data[:, 0], y=data[:, 1], color=cdict[cname], s=1, alpha=0.1, rasterized=True)
            axes[1, rule_idx].legend(handles=[plt.Line2D([0], [0], color=cdict[cname], lw=4, label=rule.title())], loc='upper left')


            axes[1, rule_idx].set_xlabel('X saccade (movement steps, centered on next line)')
            axes[1, rule_idx].set_ylabel('Y saccade (movement steps, centered on next line)')

            # axes[0, rule_idx].set_title('Steps where agent draws')
            axes[0, rule_idx].set_xlim(-.25, .25)
            axes[0, rule_idx].set_ylim(-.25, .25)
            axes[0, rule_idx].axvline(0, color='black', linestyle='--', lw=1)
            axes[0, rule_idx].axhline(0, color='black', linestyle='--', lw=1)
            axes[0, rule_idx].set_aspect('equal')
            axes[1, rule_idx].set_title('Steps where agent does not draw')
            axes[1, rule_idx].set_xlim(-.5, .5)
            axes[1, rule_idx].set_ylim(-.5, .5)
            axes[1, rule_idx].axvline(0, color='black', linestyle='--', lw=1)
            axes[1, rule_idx].axhline(0, color='black', linestyle='--', lw=1)
            axes[1, rule_idx].set_aspect('equal')

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/kde_saccades_comparisons_across_rules_frame_oracle_action.pdf')

    # TODO: same, but now all rules together, split by seed (need to tinker with load_all_seeds_results to get the correct shape for that)
    def saccade_split_by_seed():
        condition = 'only_four'
        n_seeds = 4
        all_rewards = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/cumulated_rewards.npy', out_shape=(n_seeds, -1))
        all_times = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/times.npy', out_shape=(n_seeds, -1))
        all_rules = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/rules.npy', out_shape=(n_seeds, -1))
        all_saccades = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_saccades.npy', out_shape=(n_seeds, -1, 3))
        all_homings = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_homings.npy', out_shape=(n_seeds, -1, 3))
        all_actions = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_actions.npy', out_shape=(n_seeds, -1, 3))
        all_oracle_saccades = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_oracle_saccades.npy', out_shape=(n_seeds, -1, 2))
        all_oracle_actions = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_oracle_actions.npy', out_shape=(n_seeds, -1, 3))

        

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].set_title('Drawing steps (Candidate)')
        axes[0].set_xlabel('X saccade (drawing steps)')
        axes[0].set_ylabel('Y saccade (drawing steps)')
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].axvline(0, color='black', linestyle='--')
        axes[0].set_xlim(-0.25, 0.25)
        axes[0].set_ylim(-0.25, 0.25)

        axes[1].set_title('Movement steps (Candidate)')
        axes[1].set_xlabel('X saccade (movement steps, centered on next line)')
        axes[1].set_ylabel('Y saccade (movement steps, centered on next line)')
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].axvline(0, color='black', linestyle='--')
        axes[1].set_xlim(-0.35, 0.35)
        axes[1].set_ylim(-0.35, 0.35)


        palette = sns.color_palette("deep", n_colors=n_seeds)

        for seed in range(n_seeds):

            # For this plot, all seeds at once so no care
            actions = all_actions[seed]
            saccades = all_saccades[seed]
            oracle_saccades = all_oracle_saccades[seed]

            is_drawing = (actions[:, 2] >= 0) & (np.random.rand(len(actions)) < .05)
            is_not_drawing = (actions[:, 2] < 0) & (np.random.rand(len(actions)) < .05)

            data = saccades[is_drawing, :2]  # Saccades in a frame centered around the end of the line (up to some noise)
            sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[0], color=palette[seed], fill=True, levels=5, alpha=0.75)

            data = saccades[is_not_drawing, :2] - oracle_saccades[is_not_drawing, :2] # Saccades in a frame centered around the end of the line (up to some noise)
            sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[1], color=palette[seed], fill=True, levels=5, alpha=0.75)

        axes[0].legend(handles=[plt.Line2D([0], [0], color=palette[seed], lw=4, label=f"Seed {seed}, all rules together") for seed in range(n_seeds)], loc='upper left')
        axes[1].legend(handles=[plt.Line2D([0], [0], color=palette[seed], lw=4, label=f"Seed {seed}, all rules together") for seed in range(n_seeds)], loc='upper left')      

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/saccades_comparison_between_seeds.pdf')


        # Same, but first do a 2-means clustering for the drawing steps
        from sklearn.cluster import KMeans

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        axes[0].set_title('Drawing steps (All cardinal rules)')
        axes[0].set_xlabel('X saccade (drawing steps)')
        axes[0].set_ylabel('Y saccade (drawing steps)')
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].axvline(0, color='black', linestyle='--')
        axes[0].set_xlim(-0.25, 0.25)
        axes[0].set_ylim(-0.25, 0.25)


        axes[1].set_title('Drawing steps (Closest rule)')
        axes[1].set_xlabel('X saccade (drawing steps)')
        axes[1].set_ylabel('Y saccade (drawing steps)')
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].axvline(0, color='black', linestyle='--')
        axes[1].set_xlim(-0.25, 0.25)
        axes[1].set_ylim(-0.25, 0.25)

        axes[2].set_title('Movement steps (All rules)')
        axes[2].set_xlabel('X saccade (movement steps, centered on next line)')
        axes[2].set_ylabel('Y saccade (movement steps, centered on next line)')
        axes[2].axhline(0, color='black', linestyle='--')
        axes[2].axvline(0, color='black', linestyle='--')
        axes[2].set_xlim(-0.35, 0.35)
        axes[2].set_ylim(-0.35, 0.35)


        palette = sns.color_palette("deep", n_colors=n_seeds)
        cluster_sizes = []

        for seed in range(n_seeds):
            # For this plot, all seeds at once so no care
            actions = all_actions[seed]
            saccades = all_saccades[seed]
            oracle_saccades = all_oracle_saccades[seed]
            rules = all_rules[seed]

            is_not_drawing = (actions[:, 2] < 0) & (np.random.rand(len(actions)) < .05)

            # Need more care here than for other plots because we split by rule AFTER the subsampling instead of before.
            random_selector_draw = np.random.rand(len(actions)) < .05
            is_drawing = (actions[:, 2] >= 0) & random_selector_draw
            is_closest_rule = (rules == 'closest') & random_selector_draw

            print(np.mean(rules == 'closest'))


            # All cardinal rules
            data = saccades[is_drawing & ~is_closest_rule, :2]  
            # model = KMeans(n_clusters=2, n_init='auto')
            # labels = model.fit_predict(data)
            # cluster_sizes.append(np.sum(labels==0))
            # for cluster_id in range(2):
            sns.kdeplot(x=data[:,0], y=data[:,1], ax=axes[0], color=palette[seed], fill=True, levels=5, alpha=0.75)

            # Further split between closest and other rules
            data = saccades[is_drawing & is_closest_rule, :2]  
            model = KMeans(n_clusters=2, n_init='auto')
            labels = model.fit_predict(data)
            cluster_sizes.append(int(100*np.round(np.mean(labels==0), 2)))
            for cluster_id in range(2):
                sns.kdeplot(x=data[labels==cluster_id, 0], y=data[labels==cluster_id, 1], ax=axes[1], color=palette[seed], fill=True, levels=5, alpha=0.75)


            data = saccades[is_not_drawing, :2] - oracle_saccades[is_not_drawing, :2] # Saccades in a frame centered around the end of the line (up to some noise)
            sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[2], color=palette[seed], fill=True, levels=5, alpha=0.75)

        axes[0].legend(handles=[plt.Line2D([0], [0], color=palette[seed], lw=4, label=f"Seed {seed}, all cardinal rules") for seed in range(n_seeds)], loc='upper left')
        axes[1].legend(handles=[plt.Line2D([0], [0], color=palette[seed], lw=4, label=f"Seed {seed}, closest rule, clusters are {cs}/{100-cs} \%") for seed, cs in enumerate(cluster_sizes)], loc='upper left')      
        axes[0].legend(handles=[plt.Line2D([0], [0], color=palette[seed], lw=4, label=f"Seed {seed}, all rules") for seed in range(n_seeds)], loc='upper left')

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/saccades_comparison_between_seeds_with_clustering_for_drawing_steps.pdf')


    # TODO: add a plot where we look at each direction, but split by line identity (can be inferred from oracle actions)




    # Expectation (based on previous anecdotal observations):
    # Line identity does not matter, seed does !
    # Notably, all four give different centers, but they are all in the expected direction


    # That's where we actually call our plots


    # compare_to_baselines()
    # compare_completion_rate_across_rules()
    # compare_reward_envelopes()
    # compare_saccades_across_rules()
    saccade_split_by_seed()
    








if __name__ == '__main__':
    do_comparisons()

