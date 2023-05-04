# This will require a full rerun of generalization lefttoright, as now we have added the borders so it would be unfair to test models trained without borders.
# Also, probably should remove "four_or_less experiments", since we can interpolate it's not very relevant.

import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import seaborn as sns
from scipy.stats import ks_2samp
from itertools import product
import os


# wdir = '/scratch/atf6569/saccade_drawing/'
wdir = '/home/arnaud/Scratch/saccade_drawing/'


OUT_DIR = wdir + 'generalization_lefttoright/'   


def do_comparisons():
    p = sns.color_palette('muted')
    # p = sns.color_palette()
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

    # Instead of plotting, all vs all, maybe do something smarter where we choose a few relevant comparisons

    # Might want to rework all of this !!
    # New version: each plot is a function
    # Summary figure will be done by stitching the results in Inkscape, will allow more control over the layout


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

        if out_shape is None:
            return all_seeds_results.flatten()
        elif all_seeds_results.shape != out_shape:
            return np.array(all_seeds_results).reshape(out_shape)
        else:
            return all_seeds_results


    def compare_on_four_lines():
        # Use the oracle as our reference for "perfection"
        # No real need for "random" reference, it's a delta at 0, use it only in the very first plot
        random = load_all_seeds_results(OUT_DIR + 'results/random__cond__only_four/{}/reciprocal_overlaps.npy')
        oracle = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_ablated = load_all_seeds_results(OUT_DIR + 'results/ablated__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_ablated_big = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__only_four/{}/reciprocal_overlaps.npy')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        ax.set_title('Trained model vs baselines')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1]
        ax.set_title('Trained model vs ablated models')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated, ax=ax, label='Ablated', stat='probability', bins=40, color=cdict['pink'])
        sns.histplot(candidate_ablated_big, ax=ax, label='Big ablated', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[2]
        ax.set_title('Trained model vs unconstrained saccade')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade, ax=ax, label='Supervised saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/four_lines_comparisons.pdf', dpi=300)

    def compare_on_interpolation():
        # Now that we know unconstrained saccades works better, use it as our "candidate"
        random_two = load_all_seeds_results(OUT_DIR + 'results/random__cond__only_two/{}/reciprocal_overlaps.npy')
        oracle_two = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__only_two/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade_two = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__only_two/{}/reciprocal_overlaps.npy')
        candidate_ablated_big_two = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__only_two/{}/reciprocal_overlaps.npy')
        candidate_two = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__only_two/{}/reciprocal_overlaps.npy')

        # Same, but with three instead of two
        random_three = load_all_seeds_results(OUT_DIR + 'results/random__cond__only_three/{}/reciprocal_overlaps.npy')
        oracle_three = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__only_three/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade_three = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__only_three/{}/reciprocal_overlaps.npy')
        candidate_ablated_big_three = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__only_three/{}/reciprocal_overlaps.npy')
        candidate_three = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__only_three/{}/reciprocal_overlaps.npy')

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # First, two lines interpolation
        ax = axes[0,0]
        ax.set_title('Trained model vs baselines (2 lines)')
        sns.histplot(oracle_two, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random_two, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate_two, ax=ax, label='Candidate (trained 4)', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[0,1]
        ax.set_title('Trained model vs ablated models (2 lines)')
        sns.histplot(oracle_two, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_two, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big_two, ax=ax, label='Ablated, big', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[0,2]
        ax.set_title('Trained model vs unconstrained saccade (2 lines)')
        sns.histplot(oracle_two, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_two, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade_two, ax=ax, label='With supervised saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        # Then, three lines interpolation
        ax = axes[1,0]
        ax.set_title('Trained model vs baselines (3 lines)')
        sns.histplot(oracle_three, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random_three, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate_three, ax=ax, label='Candidate (trained 4)', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1,1]
        ax.set_title('Trained model vs ablated models (3 lines)')
        sns.histplot(oracle_three, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_three, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big_three, ax=ax, label='Ablated, big', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1,2]
        ax.set_title('Trained model vs unconstrained saccade (3 lines)')
        sns.histplot(oracle_three, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_three, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade_three, ax=ax, label='With supervised saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/interpolation_comparisons.pdf', dpi=300)


    def compare_on_extrapolation():
        # Test on 5/6 for extrapolation
        random = load_all_seeds_results(OUT_DIR + 'results/random__cond__five_to_six/{}/reciprocal_overlaps.npy')
        oracle = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__five_to_six/{}/reciprocal_overlaps.npy')
        candidate = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__five_to_six/{}/reciprocal_overlaps.npy')
        candidate_ablated_big = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__five_to_six/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__five_to_six/{}/reciprocal_overlaps.npy')

        # Same, but with mirrored positions so even harder !
        random_mirror = load_all_seeds_results(OUT_DIR + 'results/random__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        oracle_mirror = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        candidate_mirror = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        candidate_ablated_big_mirror = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade_mirror = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')


        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        ax = axes[0, 0]
        ax.set_title('Trained model vs baselines (5/6 lines)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[0, 1]
        ax.set_title('Trained model vs ablated models (5/6 lines)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big, ax=ax, label='Big ablated', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[0, 2]
        ax.set_title('Trained model vs unconstrained saccade (5/6 lines)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade, ax=ax, label='Supervised saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1, 0]
        ax.set_title('Trained model vs baselines (5/6, mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random_mirror, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1, 1]
        ax.set_title('Trained model vs ablated models (5/6, mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big_mirror, ax=ax, label='Big ablated', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1, 2]
        ax.set_title('Trained model vs unconstrained saccade (5/6, mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade_mirror, ax=ax, label='Supervised saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/extrapolation_comparisons.pdf', dpi=300)

    def compare_reward_envelopes():
        # Do it on 8 lines, very long timeout 
        # For "all_rules", we might want to split rules, check if "closest" has some specificities
        # eg it always completes all lines if you leave it enough time ?
        random = load_all_seeds_results(OUT_DIR + 'results/random__cond__six_mirrored_timeout_thirty/{}/cumulated_rewards.npy')
        random_times = load_all_seeds_results(OUT_DIR + 'results/random__cond__six_mirrored_timeout_thirty/{}/times.npy')

        oracle = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__six_mirrored_timeout_thirty/{}/cumulated_rewards.npy')
        oracle_times = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__six_mirrored_timeout_thirty/{}/times.npy')

        candidate = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__six_mirrored_timeout_thirty/{}/cumulated_rewards.npy')
        candidate_times = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__six_mirrored_timeout_thirty/{}/times.npy')

        big_ablated = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__six_mirrored_timeout_thirty/{}/cumulated_rewards.npy')
        big_ablated_times = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__six_mirrored_timeout_thirty/{}/times.npy')

        saccade_constrained = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__six_mirrored_timeout_thirty/{}/cumulated_rewards.npy')
        saccade_constrained_times = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__six_mirrored_timeout_thirty/{}/times.npy')


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
        all_names = ['Random', 'Oracle', 'Candidate', 'Big ablated', 'Saccade constrained']
        all_data = [random, oracle, candidate]
        all_times = [random_times, oracle_times, candidate_times]
        all_colors = ['brown', 'blue', 'green', 'purple', 'orange']
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

    def compare_saccades_to_supervised(n_seeds=4):
        condition = 'only_four'

        loading_dict = {}

        rewards = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/cumulated_rewards.npy', out_shape=(n_seeds, -1))
        times = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/times.npy', out_shape=(n_seeds, -1))
        rules = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/rules.npy', out_shape=(n_seeds, -1))
        saccades = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_saccades.npy', out_shape=(n_seeds, -1, 3))
        homings = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_homings.npy', out_shape=(n_seeds, -1, 3))
        actions = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_actions.npy', out_shape=(n_seeds, -1, 3))
        oracle_saccades = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_oracle_saccades.npy', out_shape=(n_seeds, -1, 2))
        oracle_actions = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__' + condition + '/{}/all_oracle_actions.npy', out_shape=(n_seeds, -1, 3))

        loading_dict.update({
            'candidate': {
                'rewards': rewards,
                'times': times,
                'rules': rules,
                'saccades': saccades,
                'homings': homings,
                'actions': actions,
                'oracle_saccades': oracle_saccades,
                'oracle_actions': oracle_actions
            }
            })

        supervised_rewards = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__' + condition + '/{}/cumulated_rewards.npy', out_shape=(n_seeds, -1))
        supervised_times = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__' + condition + '/{}/times.npy', out_shape=(n_seeds, -1))
        supervised_rules = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__' + condition + '/{}/rules.npy', out_shape=(n_seeds, -1))
        supervised_saccades = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__' + condition + '/{}/all_saccades.npy', out_shape=(n_seeds, -1, 3))
        supervised_homings = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__' + condition + '/{}/all_homings.npy', out_shape=(n_seeds, -1, 3))
        supervised_actions = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__' + condition + '/{}/all_actions.npy', out_shape=(n_seeds, -1, 3))
        supervised_oracle_saccades = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__' + condition + '/{}/all_oracle_saccades.npy', out_shape=(n_seeds, -1, 2))
        supervised_oracle_actions = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__' + condition + '/{}/all_oracle_actions.npy', out_shape=(n_seeds, -1, 3))


        loading_dict.update({
            'supervised': {
                'rewards': supervised_rewards,
                'times': supervised_times,
                'rules': supervised_rules,
                'saccades': supervised_saccades,
                'homings': supervised_homings,
                'actions': supervised_actions,
                'oracle_saccades': supervised_oracle_saccades,
                'oracle_actions': supervised_oracle_actions
            }
            })




        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].set_title('Drawing steps')
        axes[0].set_xlabel('X saccade (drawing steps)')
        axes[0].set_ylabel('Y saccade (drawing steps)')
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].axvline(0, color='black', linestyle='--')
        axes[0].set_xlim(-0.25, 0.25)
        axes[0].set_ylim(-0.25, 0.25)

        axes[1].set_title('Movement steps')
        axes[1].set_xlabel('X saccade (movement steps, centered on next line)')
        axes[1].set_ylabel('Y saccade (movement steps, centered on next line)')
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].axvline(0, color='black', linestyle='--')
        axes[1].set_xlim(-0.25, 0.25)
        axes[1].set_ylim(-0.25, 0.25)

        for model_idx, model_name, cname in zip(range(2), ['candidate', 'supervised'], ['green', 'orange']):
            # Split between movements where the agent draws, and ones where it does not
            actions = loading_dict[model_name]['actions']
            saccades = loading_dict[model_name]['saccades']
            oracle_saccades = loading_dict[model_name]['oracle_saccades']

            # For this plot, all seeds at once so no care
            actions = actions.reshape(-1, 3)
            saccades = saccades.reshape(-1, 3)
            oracle_saccades = oracle_saccades.reshape(-1, 2)

            is_drawing = (actions[:, 2] >= 0) 
            is_not_drawing = (actions[:, 2] < 0)

            data = saccades[is_drawing, :2]  # Saccades in a frame centered around the end of the line (up to some noise)

            # sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[0], color=cdict[cname], fill=True)
            axes[0].scatter(data[:, 0], data[:, 1], color=cdict[cname], s=1, alpha=.01, rasterized=True)
        
            data = saccades[is_not_drawing, :2] - oracle_saccades[is_not_drawing, :2] # Saccades in a frame centered around the end of the line (up to some noise)
            # sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[1], color=cdict[cname], fill=True)
            axes[1].scatter(data[:, 0], data[:, 1], color=cdict[cname], s=1, alpha=.01, rasterized=True)


        axes[0].legend(handles=[plt.Line2D([0], [0], color=cdict[cname], lw=4, label=model_name.title()) for cname, model_name in zip(['green', 'orange'], ['Candidate', 'Supervised'])], loc='upper left')
        axes[1].legend(handles=[plt.Line2D([0], [0], color=cdict[cname], lw=4, label=model_name.title()) for cname, model_name in zip(['green', 'orange'], ['Candidate', 'Supervised'])], loc='upper left')      

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/scatter_saccades_comparison_vs_constrained.pdf')

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].set_title('Drawing steps')
        axes[0].set_xlabel('X saccade (drawing steps)')
        axes[0].set_ylabel('Y saccade (drawing steps)')
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].axvline(0, color='black', linestyle='--')
        axes[0].set_xlim(-0.25, 0.25)
        axes[0].set_ylim(-0.25, 0.25)

        axes[1].set_title('Movement steps')
        axes[1].set_xlabel('X saccade (movement steps, centered on next line)')
        axes[1].set_ylabel('Y saccade (movement steps, centered on next line)')
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].axvline(0, color='black', linestyle='--')
        axes[1].set_xlim(-0.35, 0.35)
        axes[1].set_ylim(-0.35, 0.35)
        for model_idx, model_name, cname in zip(range(2), ['candidate', 'supervised'], ['green', 'orange']):
            # Split between movements where the agent draws, and ones where it does not
            actions = loading_dict[model_name]['actions']
            saccades = loading_dict[model_name]['saccades']
            oracle_saccades = loading_dict[model_name]['oracle_saccades']
            # For this plot, all seeds at once so no care
            actions = actions.reshape(-1, 3)
            saccades = saccades.reshape(-1, 3)
            oracle_saccades = oracle_saccades.reshape(-1, 2)
            is_drawing = (actions[:, 2] >= 0) & (np.random.rand(len(actions)) < .05)
            is_not_drawing = (actions[:, 2] < 0) & (np.random.rand(len(actions)) < .05)

            data = saccades[is_drawing, :2]  # Saccades in a frame centered around the end of the line (up to some noise)
            sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[0], color=cdict[cname], fill=True)
            data = saccades[is_not_drawing, :2] - oracle_saccades[is_not_drawing, :2] # Saccades in a frame centered around the end of the line (up to some noise)
            sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[1], color=cdict[cname], fill=True)

        axes[0].legend(handles=[plt.Line2D([0], [0], color=cdict[cname], lw=4, label=model_name.title()) for cname, model_name in zip(['green', 'orange'], ['Candidate', 'Supervised'])], loc='upper left')
        axes[1].legend(handles=[plt.Line2D([0], [0], color=cdict[cname], lw=4, label=model_name.title()) for cname, model_name in zip(['green', 'orange'], ['Candidate', 'Supervised'])], loc='upper left')      

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/kde_saccades_comparison_vs_constrained.pdf')

        # Do one with seeds split
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes[0, 0].set_title('Drawing steps (Candidate)')
        axes[0, 0].set_xlabel('X saccade (drawing steps)')
        axes[0, 0].set_ylabel('Y saccade (drawing steps)')
        axes[0, 0].axhline(0, color='black', linestyle='--')
        axes[0, 0].axvline(0, color='black', linestyle='--')
        axes[0, 0].set_xlim(-0.25, 0.25)
        axes[0, 0].set_ylim(-0.25, 0.25)

        axes[0, 1].set_title('Drawing steps (Supervised)')
        axes[0, 1].set_xlabel('X saccade (drawing steps)')
        axes[0, 1].set_ylabel('Y saccade (drawing steps)')
        axes[0, 1].axhline(0, color='black', linestyle='--')
        axes[0, 1].axvline(0, color='black', linestyle='--')
        axes[0, 1].set_xlim(-0.25, 0.25)
        axes[0, 1].set_ylim(-0.25, 0.25)

        axes[1, 0].set_title('Movement steps (Candidate)')
        axes[1, 0].set_xlabel('X saccade (movement steps, centered on next line)')
        axes[1, 0].set_ylabel('Y saccade (movement steps, centered on next line)')
        axes[1, 0].axhline(0, color='black', linestyle='--')
        axes[1, 0].axvline(0, color='black', linestyle='--')
        axes[1, 0].set_xlim(-0.35, 0.35)
        axes[1, 0].set_ylim(-0.35, 0.35)

        axes[1, 1].set_title('Movement steps (Supervised)')
        axes[1, 1].set_xlabel('X saccade (movement steps, centered on next line)')
        axes[1, 1].set_ylabel('Y saccade (movement steps, centered on next line)')
        axes[1, 1].axhline(0, color='black', linestyle='--')
        axes[1, 1].axvline(0, color='black', linestyle='--')
        axes[1, 1].set_xlim(-0.35, 0.35)
        axes[1, 1].set_ylim(-0.35, 0.35)

        # green_variations = sns.color_palette('crest', n_colors=n_seeds)
        # red_variations = sns.color_palette('flare', n_colors=n_seeds)

        green_variations = sns.color_palette(n_colors=n_seeds)
        red_variations = green_variations

        for model_idx, model_name, seed_colors in zip(range(2), ['candidate', 'supervised'], [green_variations, red_variations]):
            for seed in range(n_seeds):
                print(seed)
                # Split between movements where the agent draws, and ones where it does not
                actions = loading_dict[model_name]['actions']
                saccades = loading_dict[model_name]['saccades']
                oracle_saccades = loading_dict[model_name]['oracle_saccades']
                # For this plot, all seeds at once so no care
                actions = actions[seed]
                saccades = saccades[seed]
                oracle_saccades = oracle_saccades[seed]

                is_drawing = (actions[:, 2] >= 0) & (np.random.rand(len(actions)) < .05)
                is_not_drawing = (actions[:, 2] < 0) & (np.random.rand(len(actions)) < .05)

                data = saccades[is_drawing, :2]  # Saccades in a frame centered around the end of the line (up to some noise)
                sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[0, model_idx], color=seed_colors[seed], fill=True, levels=5, alpha=0.5)

                data = saccades[is_not_drawing, :2] - oracle_saccades[is_not_drawing, :2] # Saccades in a frame centered around the end of the line (up to some noise)
                sns.kdeplot(x=data[:, 0], y=data[:, 1], ax=axes[1, model_idx], color=seed_colors[seed], fill=True, levels=5, alpha=0.5)

            axes[0, model_idx].legend(handles=[plt.Line2D([0], [0], color=seed_colors[seed], lw=4, label=f"Seed {seed}") for seed in range(n_seeds)], loc='upper left')
            axes[1, model_idx].legend(handles=[plt.Line2D([0], [0], color=seed_colors[seed], lw=4, label=f"Seed {seed}") for seed in range(n_seeds)], loc='upper left')      

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/saccades_comparison_between_seeds.pdf')


    compare_on_four_lines()
    compare_on_interpolation()
    compare_on_extrapolation()
    compare_reward_envelopes()
    compare_saccades_to_supervised()
    


    





if __name__ == '__main__':
    do_comparisons()


