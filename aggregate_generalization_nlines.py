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
# wdir = '/home/arnaud/Scratch/saccade_drawing/'
# wdir = '/home/arnaud/Scratch/saccade_drawing_out/'


# wdir = '/home/arnaud/Scratch/saccade_drawing/'
wdir = '/scratch/atf6569/saccade_drawing/'



# OUT_DIR = wdir + 'generalization_n_lines_no_ambiguous_orderings/'
# OUT_DIR = wdir + 'n_lines_generalization/'   
OUT_DIR = wdir + 'generalization_n_lines/'   


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

    # Instead of plotting, all vs all, maybe do something smarter where we choose a few relevant comparisons

    # Might want to rework all of this !!
    # New version: each plot is a function
    # Summary figure will be done by stitching the results in Inkscape, will allow more control over the layout

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
        candidate = load_all_seeds_results(OUT_DIR + 'results/four_only__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_ablated = load_all_seeds_results(OUT_DIR + 'results/ablated_four_only__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_ablated_big = load_all_seeds_results(OUT_DIR + 'results/ablated_four_only_big_peripheral__cond__only_four/{}/reciprocal_overlaps.npy')
        candidate_unconstrained_saccade = load_all_seeds_results(OUT_DIR + 'results/four_lines_no_saccades__cond__only_four/{}/reciprocal_overlaps.npy')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        ax.set_title('Trained model vs baselines')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1]
        ax.set_title('Trained model vs ablated models')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['orange'])
        sns.histplot(candidate_ablated, ax=ax, label='With fovea ablation', stat='probability', bins=40, color=cdict['pink'])
        sns.histplot(candidate_ablated_big, ax=ax, label='With fovea ablation, bigger', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[2]
        ax.set_title('Trained model vs unconstrained saccade')
        sns.histplot(oracle, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['orange'])
        sns.histplot(candidate_unconstrained_saccade, ax=ax, label='Unconstrained saccades', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/four_lines_comparisons.pdf', dpi=300)

    def compare_on_interpolation():
        # Now that we know unconstrained saccades works better, use it as our "candidate"
        random_two = load_all_seeds_results(OUT_DIR + 'results/random__cond__only_two/{}/reciprocal_overlaps.npy')
        oracle_two = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__only_two/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade_two = load_all_seeds_results(OUT_DIR + 'results/four_only__cond__only_two/{}/reciprocal_overlaps.npy')
        candidate_ablated_big_two = load_all_seeds_results(OUT_DIR + 'results/ablated_four_only_big_peripheral__cond__only_two/{}/reciprocal_overlaps.npy')
        candidate_two = load_all_seeds_results(OUT_DIR + 'results/four_lines_no_saccades__cond__only_two/{}/reciprocal_overlaps.npy')

        # Same, but with three instead of two
        random_three = load_all_seeds_results(OUT_DIR + 'results/random__cond__only_three/{}/reciprocal_overlaps.npy')
        oracle_three = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__only_three/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade_three = load_all_seeds_results(OUT_DIR + 'results/four_only__cond__only_three/{}/reciprocal_overlaps.npy')
        candidate_ablated_big_three = load_all_seeds_results(OUT_DIR + 'results/ablated_four_only_big_peripheral__cond__only_three/{}/reciprocal_overlaps.npy')
        candidate_three = load_all_seeds_results(OUT_DIR + 'results/four_lines_no_saccades__cond__only_three/{}/reciprocal_overlaps.npy')

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
        candidate = load_all_seeds_results(OUT_DIR + 'results/four_lines_no_saccades__cond__five_to_six/{}/reciprocal_overlaps.npy')
        candidate_ablated_big = load_all_seeds_results(OUT_DIR + 'results/ablated_four_only_big_peripheral__cond__five_to_six/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade = load_all_seeds_results(OUT_DIR + 'results/four_only__cond__five_to_six/{}/reciprocal_overlaps.npy')

        # Same, but with mirrored positions so even harder !
        random_mirror = load_all_seeds_results(OUT_DIR + 'results/random__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        oracle_mirror = load_all_seeds_results(OUT_DIR + 'results/oracle__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        candidate_mirror = load_all_seeds_results(OUT_DIR + 'results/four_lines_no_saccades__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        candidate_ablated_big_mirror = load_all_seeds_results(OUT_DIR + 'results/ablated_four_only_big_peripheral__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')
        candidate_constrained_saccade_mirror = load_all_seeds_results(OUT_DIR + 'results/four_only__cond__mirrored_five_to_six/{}/reciprocal_overlaps.npy')


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
        ax.set_title('Trained model vs baselines (mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(random_mirror, ax=ax, label='Random agent', stat='probability', bins=40, color=cdict['brown'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1, 1]
        ax.set_title('Trained model vs ablated models (mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_ablated_big_mirror, ax=ax, label='Big ablated', stat='probability', bins=40, color=cdict['purple'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        ax = axes[1, 2]
        ax.set_title('Trained model vs unconstrained saccade (mirrored)')
        sns.histplot(oracle_mirror, ax=ax, label='Oracle', stat='probability', bins=40, color=cdict['blue'])
        sns.histplot(candidate_mirror, ax=ax, label='Candidate', stat='probability', bins=40, color=cdict['green'])
        sns.histplot(candidate_constrained_saccade_mirror, ax=ax, label='Supervised saccades', stat='probability', bins=40, color=cdict['orange'])
        ax.set_xlabel('Reciprocal overlap')
        ax.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR + 'results/extrapolation_comparisons.pdf', dpi=300)

    compare_on_four_lines()
    compare_on_interpolation()
    compare_on_extrapolation()
    


    # # First, show that our network deals well with 1-4, 4, 5-6
    # overlap_ref = []
    # overlap = []
    # overlap_ablated = []
    # overlap_interpolation_ref = []
    # overlap_extrapolation_ref = []
    # overlap_interpolation = []
    # overlap_extrapolation = []
    # overlap_interpolation_ablated = []
    # overlap_extrapolation_ablated = []
    # overlap_ablated_big = []
    # overlap_interpolation_ablated_big = []
    # overlap_extrapolation_ablated_big = []

    # overlap_mirrored = []

    # for seed in range(4):
    #     overlap_ref.append(np.load(OUT_DIR + f'results/four_only__cond__only_four/{seed}/reciprocal_overlaps.npy'))
    #     overlap.append(np.load(OUT_DIR + f'results/four_only__cond__only_four/{seed}/reciprocal_overlaps.npy'))
    #     overlap_ablated.append(np.load(OUT_DIR + f'results/ablated_four_only__cond__only_four/{seed}/reciprocal_overlaps.npy'))
    #     overlap_interpolation_ref.append(np.load(OUT_DIR + f'results/four_or_less__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))

    #     overlap_extrapolation_ref.append(np.load(OUT_DIR + f'results/four_only__cond__only_four/{seed}/reciprocal_overlaps.npy'))
    #     overlap_interpolation.append(np.load(OUT_DIR + f'results/four_only__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))
    #     overlap_extrapolation.append(np.load(OUT_DIR + f'results/four_only__cond__five_to_six/{seed}/reciprocal_overlaps.npy'))
    #     overlap_interpolation_ablated.append(np.load(OUT_DIR + f'results/ablated_four_only__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))
    #     overlap_extrapolation_ablated.append(np.load(OUT_DIR + f'results/ablated_four_only__cond__five_to_six/{seed}/reciprocal_overlaps.npy'))

    #     # Not done running yet, do with the four_only for now
    #     # overlap_ablated_big.append(np.load(OUT_DIR + f'results/ablated_four_or_less_big_peripheral__cond__only_four/{seed}/reciprocal_overlaps.npy'))
    #     # overlap_interpolation_ablated_big.append(np.load(OUT_DIR + f'results/ablated_four_or_less_big_peripheral__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))
    #     # overlap_extrapolation_ablated_big.append(np.load(OUT_DIR + f'results/ablated_four_or_less_big_peripheral__cond__five_to_six/{seed}/reciprocal_overlaps.npy'))

    #     overlap_ablated_big.append(np.load(OUT_DIR + f'results/ablated_four_only_big_peripheral__cond__only_four/{seed}/reciprocal_overlaps.npy'))
    #     overlap_interpolation_ablated_big.append(np.load(OUT_DIR + f'results/ablated_four_only_big_peripheral__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))
    #     overlap_extrapolation_ablated_big.append(np.load(OUT_DIR + f'results/ablated_four_only_big_peripheral__cond__five_to_six/{seed}/reciprocal_overlaps.npy'))


    #     overlap_mirrored.append(np.load(OUT_DIR + f'results/four_only__cond__mirrored_one_to_six/{seed}/reciprocal_overlaps.npy'))


    # overlap_ref = np.concatenate(overlap_ref).flatten()
    # overlap = np.concatenate(overlap).flatten()
    # overlap_ablated = np.concatenate(overlap_ablated).flatten()
    # overlap_interpolation_ref = np.concatenate(overlap_interpolation_ref).flatten()
    # overlap_extrapolation_ref = np.concatenate(overlap_extrapolation_ref).flatten()
    # overlap_interpolation = np.concatenate(overlap_interpolation).flatten()
    # overlap_extrapolation = np.concatenate(overlap_extrapolation).flatten()
    # overlap_interpolation_ablated = np.concatenate(overlap_interpolation_ablated).flatten()
    # overlap_extrapolation_ablated = np.concatenate(overlap_extrapolation_ablated).flatten()
    # overlap_ablated_big = np.concatenate(overlap_ablated_big).flatten()
    # overlap_interpolation_ablated_big = np.concatenate(overlap_interpolation_ablated_big).flatten()
    # overlap_extrapolation_ablated_big = np.concatenate(overlap_extrapolation_ablated_big).flatten()
    # overlap_mirrored = np.concatenate(overlap_mirrored).flatten()

    # fig, axes = plt.subplots(2,3, figsize=(17,8))

    # ax = axes[0,0]
    # sns.histplot(overlap_ref, ax=ax, label='Reference (4, trained 4)', stat='probability', bins=40)
    # sns.histplot(overlap_ablated, ax=ax, label='Ablated (4, trained 4)', stat='probability', bins=40)
    # sns.histplot(overlap_ablated_big, ax=ax, label='Big Ablated (4, trained 1-4)', stat='probability', bins=40)
    # ax.set_xlabel('Reciprocal overlap at trajectory timeout')
    # ax.set_title('Full vs ablated network')
    # ax.legend()

    # ax = axes[1,0]
    # sns.histplot(overlap_extrapolation_ref, ax=ax, label='Reference (4, trained 4)', stat='probability', bins=40)
    # sns.histplot(overlap_extrapolation, ax=ax, label='Same spawns (5-6, trained 4)', stat='probability', bins=40)
    # sns.histplot(overlap_mirrored, ax=ax, label='New spawns (5-6, trained 4)', stat='probability', bins=40)
    # ax.set_xlabel('Reciprocal overlap at trajectory timeout')

    # ax.set_title('Mirroring')
    # ax.legend()

    # ax = axes[0,1]
    # sns.histplot(overlap_interpolation_ref, ax=ax, label='Reference (1-4, trained 1-4)', stat='probability', bins=40)
    # sns.histplot(overlap_interpolation, ax=ax, label='Interpolation (1-4, trained 4)', stat='probability', bins=40)
    # ax.set_xlabel('Reciprocal overlap at trajectory timeout')
    # ax.set_title('Interpolation using full network')
    # ax.legend()

    # ax = axes[1,1]
    # sns.histplot(overlap_interpolation_ref, ax=ax, label='Reference (1-4, trained 1-4)', stat='probability', bins=40)
    # sns.histplot(overlap_interpolation_ablated, ax=ax, label='Ablated (1-4, trained 4)', stat='probability', bins=40)
    # sns.histplot(overlap_interpolation_ablated_big, ax=ax, label='Big Ablated (1-4, trained 1-4)', stat='probability', bins=40)
    # ax.set_xlabel('Reciprocal overlap at trajectory timeout')
    # ax.set_title('Interpolation using ablated network')
    # ax.legend()

    # ax = axes[0,2]
    # sns.histplot(overlap_extrapolation_ref, ax=ax, label='Reference (4, trained 4)', stat='probability', bins=40)
    # sns.histplot(overlap_extrapolation, ax=ax, label='Extrapolation (5-6, trained 4)', stat='probability', bins=40)
    # ax.set_xlabel('Reciprocal overlap at trajectory timeout')
    # ax.set_title('Extrapolation using full network')
    # ax.legend()

    # ax = axes[1,2]
    # sns.histplot(overlap_extrapolation_ref, ax=ax, label='Reference (4, trained 4)', stat='probability', bins=40)
    # sns.histplot(overlap_extrapolation_ablated, ax=ax, label='Ablated (5-6, trained 4)', stat='probability', bins=40)
    # sns.histplot(overlap_extrapolation_ablated_big, ax=ax, label='Big Ablated (5-6, trained 1-4)', stat='probability', bins=40)
    # ax.set_xlabel('Reciprocal overlap at trajectory timeout')
    # ax.set_title('Extrapolation using ablated network')
    # ax.legend()

    # fig.suptitle('Generalization summary')
    # fig.tight_layout()
    # fig.savefig(OUT_DIR + 'results/generalization_summary.pdf')
    # plt.close(fig)





if __name__ == '__main__':
    do_comparisons()


# def compare_distributions(d0, d1, labels, savepath, n_perms=1000):
#     os.makedirs(savepath, exist_ok=True)

#     k2_res = ks_2samp(d0, d1) # Kolmogorov-Smirnov two-samples test, two-sided and asymptotic
#     k2 = (k2_res.statistic, k2_res.pvalue)

#     fig, ax = plt.subplots()
#     sns.histplot(d0, ax=ax, label=labels[0])
#     sns.histplot(d1, ax=ax, label=labels[1])
#     ax.legend()
#     ax.set_title('KS test: D = {:.3f}, p = {:.3f}'.format(k2[0], k2[1]))
#     fig.savefig(savepath + 'distributions.pdf')
#     plt.close(fig)

#     # Do some kind of bootstrap to see whether the difference is really relevant
#     perm_stats = []
#     perm_p = []
#     plop = np.concatenate([d0, d1])
#     np_random = np.random.default_rng()
#     for _ in range(n_perms):
#         perm = np_random.choice(plop, size=len(plop), replace=False)
#         d0_perm = perm[:len(d0)]
#         d1_perm = perm[len(d0):]
#         # print(d0_perm[:5], d1_perm[:5])
#         k2_perm_res = ks_2samp(d0_perm, d1_perm)
#         perm_stats.append(k2_perm_res.statistic)
#         perm_p.append(k2_perm_res.pvalue)

#     perm_stats = np.array(perm_stats)
#     perm_p = np.array(perm_p)

#     fig, ax = plt.subplots()
#     sns.histplot(perm_stats, ax=ax, label='Randomized populations')
#     ax.axvline(k2[0], color='r', label='Initial populations')
#     ax.legend()
#     fig.savefig(savepath + 'distributions_randomized.pdf')

#     plt.close('all')

#     # for var in ['errors_homing', 'tprs', 'reciprocal_overlaps']:
#     #     for testing_condition in ['only_four', 'four_or_less', 'five_to_six']:
#     #         # for net1, net2 in product(['only_four', 'ablated_four'], [['only_four', 'ablated_four']]):
#     #         for net1, net2 in product(['only_four', 'four_or_less'], ['only_four', 'four_or_less']):
#     #             if net1 == net2:
#     #                 continue
#     #             savepath = OUT_DIR + f'comparisons/{var}_{testing_condition}__agent_{net1}_vs_{net2}/'
#     #             load_from_template = OUT_DIR + 'results/{}' + '__cond__' + testing_condition + '/{}.npy'
#     #             d0 = np.load(load_from_template.format(net1, var))
#     #             d1 = np.load(load_from_template.format(net2, var))
#     #             # Note that this test is not totally symmetrical, so it's ok to call it with both orderings
#     #             compare_distributions(d0, d1, [net1, net2], savepath)
