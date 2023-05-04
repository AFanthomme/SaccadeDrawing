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


OUT_DIR = wdir + 'generalization_closest/'   


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

    def load_all_seeds_results(template_path, n_seeds=4):
        # template_path should be something like 'results/oracle__cond__only_four/{}/reciprocal_overlaps.npy'
        all_seeds_results = []
        for seed in range(n_seeds):
            all_seeds_results.append(np.load(template_path.format(seed)))
        all_seeds_results = np.array(all_seeds_results).flatten()
        return all_seeds_results


    def compare_on_four_lines():
        # Use the oracle as our reference for "perfection"
        # No real need for "random" reference, it's a delta at 0, use it only in the very first plot
        random = load_all_seeds_results(OUT_DIR + 'results/random__cond__only_four/{}/reciprocal_overlaps.npy')
        oracle = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_ablated = load_all_seeds_results(OUT_DIR + 'results/ablated__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_ablated_big = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate = load_all_seeds_results(OUT_DIR + 'results/candidate__cond__only_four/{}/reciprocal_overlaps.npy')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        ax.set_title('Candidate vs baselines')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1]
        ax.set_title('Candidate vs ablated models')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated, ax=ax, label='With fovea ablation', stat='probability', bins=40, color=cdict['pink'])
        sns.histplot(candidate_ablated_big, ax=ax, label='With fovea ablation, bigger', stat='probability', bins=40, color=cdict['purple'])
        sns.histplot(candidate_constrained_saccade, ax=ax, label='Constrained saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[2]
        ax.set_title('Candidate vs unconstrained saccade')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade, ax=ax, label='Constrained saccades', stat='probability', bins=40, color=cdict['orange'])
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
        ax.set_title('Candidate vs baselines (2 lines)')
        sns.histplot(oracle_two, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random_two, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate_two, ax=ax, label='Candidate (trained 4)', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[0,1]
        ax.set_title('Candidate vs ablated models (2 lines)')
        sns.histplot(oracle_two, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_two, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big_two, ax=ax, label='Ablated, big', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[0,2]
        ax.set_title('Candidate vs unconstrained saccade (2 lines)')
        sns.histplot(oracle_two, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_two, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade_two, ax=ax, label='Constrained saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        # Then, three lines interpolation
        ax = axes[1,0]
        ax.set_title('Candidate vs baselines (3 lines)')
        sns.histplot(oracle_three, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random_three, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate_three, ax=ax, label='Candidate (trained 4)', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1,1]
        ax.set_title('Candidate vs ablated models (3 lines)')
        sns.histplot(oracle_three, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_three, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big_three, ax=ax, label='Ablated, big', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1,2]
        ax.set_title('Candidate vs unconstrained saccade (3 lines)')
        sns.histplot(oracle_three, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_three, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade_three, ax=ax, label='Constrained saccades', stat='probability', bins=40, color=cdict['orange'])
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
        ax.set_title('Candidate vs baselines (5/6 lines)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[0, 1]
        ax.set_title('Candidate vs ablated models (5/6 lines)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big, ax=ax, label='Big ablated', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[0, 2]
        ax.set_title('Candidate vs constrained saccade (5/6 lines)')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade, ax=ax, label='Constrained saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1, 0]
        ax.set_title('Candidate vs baselines (mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random_mirror, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1, 1]
        ax.set_title('Candidate vs ablated models (mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big_mirror, ax=ax, label='Big ablated', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1, 2]
        ax.set_title('Candidate vs constrained saccade (mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade_mirror, ax=ax, label='Constrained saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/extrapolation_comparisons.pdf', dpi=300)

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

        candidate_ablated_big = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__mirrored_6_timeout_30/{}/cumulated_rewards.npy')
        candidate_ablated_big_times = load_all_seeds_results(OUT_DIR + 'results/big_ablated__cond__mirrored_6_timeout_30/{}/times.npy')

        candidate_constrained_saccade = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__mirrored_6_timeout_30/{}/cumulated_rewards.npy')
        candidate_constrained_saccade_times = load_all_seeds_results(OUT_DIR + 'results/saccade_constrained__cond__mirrored_6_timeout_30/{}/times.npy')

        candidate_ablated_small = load_all_seeds_results(OUT_DIR + 'results/ablated__cond__mirrored_6_timeout_30/{}/cumulated_rewards.npy')
        candidate_ablated_small_times = load_all_seeds_results(OUT_DIR + 'results/ablated__cond__mirrored_6_timeout_30/{}/times.npy')

        def do_heatmaps(name, data, times, ax, color_strong, color_weak):
            # Two different colors needed to make the filled area readable
            # Easy to get from the seaborn colormaps
            unique_times = np.unique(times) 
            assert np.all(unique_times == np.arange(0, 30))
            assert max(data) <= 8 # 8 lines, so should get at most 8 rewards in a trajectory
            means, stds = [], []

            n_lines = int(max(data))

            all_counts = np.zeros((30, n_lines+1)) # 30 steps, 9 possibilities for cumulated reward
            

            for t in unique_times:
                filtered_data = data[times == t]
                counts = np.bincount(filtered_data.astype(int), minlength=n_lines+1)
                all_counts[t] = counts
                means.append(np.mean(filtered_data))
                stds.append(np.std(filtered_data))
            
            all_counts = all_counts.astype(float)
            probs = all_counts / np.sum(all_counts, axis=1, keepdims=True)
            
            sns.heatmap(probs.T[::-1, :], yticklabels=range(n_lines, -1, -1), linewidths=.5, annot=True, fmt='.2f', ax=ax)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Number of lines completed')
            ax.set_title(f"Using {name} agent")


        def do_avg_completion(name, data, times, ax, color_strong, color_weak):
            # Two different colors needed to make the filled area readable
            # Easy to get from the seaborn colormaps
            unique_times = np.unique(times) 
            assert np.all(unique_times == np.arange(0, 30))
            assert max(data) <= 8 # 8 lines, so should get at most 8 rewards in a trajectory
            means, stds = [], []

            n_lines = int(max(data))
            all_counts = np.zeros((30, n_lines+1)) # 30 steps, 9 possibilities for cumulated reward
            
            for t in unique_times:
                filtered_data = data[times == t]
                counts = np.bincount(filtered_data.astype(int), minlength=n_lines+1)
                all_counts[t] = counts
                means.append(np.mean(filtered_data))
                stds.append(np.std(filtered_data))

            ax.plot(unique_times, means, color=color_strong, label=name)
            ax.fill_between(unique_times, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color_weak, alpha=.3)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Mean number of lines completed')

        fig, axes = plt.subplots(5, 1, figsize=(16, 20))
        all_names = ['Random', 'Oracle', 'Candidate', 'Big ablated', 'Constrained saccade']
        all_data = [random, oracle, candidate, candidate_ablated_big, candidate_constrained_saccade]
        all_times = [random_times, oracle_times, candidate_times, candidate_ablated_big_times, candidate_constrained_saccade_times]
        all_colors = ['brown', 'blue', 'green', 'purple', 'orange', 'pink']
        for name, data, times, ax, cname in zip(all_names, all_data, all_times, axes, all_colors):
            do_heatmaps(name, data, times, ax, cdict[cname], cdict_pastel[cname])
        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/completed_lines_envelope_comparisons.pdf')


        fig, summary_ax = plt.subplots(1, 1, figsize=(8, 6))
        for name, data, times, cname in zip(all_names, all_data, all_times, all_colors):
            # Left panel will look like garbage, but right one should be ok
            do_avg_completion(name, data, times, summary_ax, cdict[cname], cdict_pastel[cname])
        summary_ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/average_completed_lines_comparisons.pdf')

    # compare_on_four_lines()
    # compare_on_interpolation()
    # compare_on_extrapolation()
    compare_reward_envelopes()
    








if __name__ == '__main__':
    do_comparisons()

