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
wdir = '/home/arnaud/Scratch/saccade_drawing_out/'

# ROOT_OUTPUT_FOLDER = wdir + 'generalization_n_lines_no_ambiguous_orderings/'
ROOT_OUTPUT_FOLDER = wdir + 'n_lines_generalization/'   


def compare_distributions(d0, d1, labels, savepath, n_perms=1000):
    os.makedirs(savepath, exist_ok=True)

    k2_res = ks_2samp(d0, d1) # Kolmogorov-Smirnov two-samples test, two-sided and asymptotic
    k2 = (k2_res.statistic, k2_res.pvalue)

    fig, ax = plt.subplots()
    sns.histplot(d0, ax=ax, label=labels[0])
    sns.histplot(d1, ax=ax, label=labels[1])
    ax.legend()
    ax.set_title('KS test: D = {:.3f}, p = {:.3f}'.format(k2[0], k2[1]))
    fig.savefig(savepath + 'distributions.pdf')
    plt.close(fig)

    # Do some kind of bootstrap to see whether the difference is really relevant
    perm_stats = []
    perm_p = []
    plop = np.concatenate([d0, d1])
    np_random = np.random.default_rng()
    for _ in range(n_perms):
        perm = np_random.choice(plop, size=len(plop), replace=False)
        d0_perm = perm[:len(d0)]
        d1_perm = perm[len(d0):]
        # print(d0_perm[:5], d1_perm[:5])
        k2_perm_res = ks_2samp(d0_perm, d1_perm)
        perm_stats.append(k2_perm_res.statistic)
        perm_p.append(k2_perm_res.pvalue)

    perm_stats = np.array(perm_stats)
    perm_p = np.array(perm_p)

    fig, ax = plt.subplots()
    sns.histplot(perm_stats, ax=ax, label='Randomized populations')
    ax.axvline(k2[0], color='r', label='Initial populations')
    ax.legend()
    fig.savefig(savepath + 'distributions_randomized.pdf')

    plt.close('all')

def do_comparisons():
    # Instead of plotting, all vs all, maybe do something smarter where we choose a few relevant comparisons

    # First, show that our network deals well with 1-4, 4, 5-6
    overlap_ref = []
    overlap = []
    overlap_ablated = []
    overlap_interpolation_ref = []
    overlap_extrapolation_ref = []
    overlap_interpolation = []
    overlap_extrapolation = []
    overlap_interpolation_ablated = []
    overlap_extrapolation_ablated = []
    overlap_ablated_big = []
    overlap_interpolation_ablated_big = []
    overlap_extrapolation_ablated_big = []

    for seed in range(4):
        overlap_ref.append(np.load(ROOT_OUTPUT_FOLDER + f'results/only_four__cond__only_four/{seed}/reciprocal_overlaps.npy'))
        overlap.append(np.load(ROOT_OUTPUT_FOLDER + f'results/only_four__cond__only_four/{seed}/reciprocal_overlaps.npy'))
        overlap_ablated.append(np.load(ROOT_OUTPUT_FOLDER + f'results/ablated_four_only__cond__only_four/{seed}/reciprocal_overlaps.npy'))
        overlap_interpolation_ref.append(np.load(ROOT_OUTPUT_FOLDER + f'results/four_or_less__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))
        overlap_extrapolation_ref.append(np.load(ROOT_OUTPUT_FOLDER + f'results/only_four__cond__only_four/{seed}/reciprocal_overlaps.npy'))
        overlap_interpolation.append(np.load(ROOT_OUTPUT_FOLDER + f'results/only_four__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))
        overlap_extrapolation.append(np.load(ROOT_OUTPUT_FOLDER + f'results/only_four__cond__five_to_six/{seed}/reciprocal_overlaps.npy'))
        overlap_interpolation_ablated.append(np.load(ROOT_OUTPUT_FOLDER + f'results/ablated_four_only__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))
        overlap_extrapolation_ablated.append(np.load(ROOT_OUTPUT_FOLDER + f'results/ablated_four_only__cond__five_to_six/{seed}/reciprocal_overlaps.npy'))
        overlap_ablated_big.append(np.load(ROOT_OUTPUT_FOLDER + f'results/ablated_four_or_less_big_peripheral__cond__only_four/{seed}/reciprocal_overlaps.npy'))
        overlap_interpolation_ablated_big.append(np.load(ROOT_OUTPUT_FOLDER + f'results/ablated_four_or_less_big_peripheral__cond__four_or_less/{seed}/reciprocal_overlaps.npy'))
        overlap_extrapolation_ablated_big.append(np.load(ROOT_OUTPUT_FOLDER + f'results/ablated_four_or_less_big_peripheral__cond__five_to_six/{seed}/reciprocal_overlaps.npy'))

    overlap_ref = np.concatenate(overlap_ref).flatten()
    overlap = np.concatenate(overlap).flatten()
    overlap_ablated = np.concatenate(overlap_ablated).flatten()
    overlap_interpolation_ref = np.concatenate(overlap_interpolation_ref).flatten()
    overlap_extrapolation_ref = np.concatenate(overlap_extrapolation_ref).flatten()
    overlap_interpolation = np.concatenate(overlap_interpolation).flatten()
    overlap_extrapolation = np.concatenate(overlap_extrapolation).flatten()
    overlap_interpolation_ablated = np.concatenate(overlap_interpolation_ablated).flatten()
    overlap_extrapolation_ablated = np.concatenate(overlap_extrapolation_ablated).flatten()
    overlap_ablated_big = np.concatenate(overlap_ablated_big).flatten()
    overlap_interpolation_ablated_big = np.concatenate(overlap_interpolation_ablated_big).flatten()
    overlap_extrapolation_ablated_big = np.concatenate(overlap_extrapolation_ablated_big).flatten()

    fig, axes = plt.subplots(2,3, figsize=(17,8))
    ax = axes[0,0]
    sns.histplot(overlap_ref, ax=ax, label='Reference (4, trained 4)', stat='probability')
    ax.set_title('Full network')
    ax.legend()

    ax = axes[1,0]
    sns.histplot(overlap_ref, ax=ax, label='Reference (4, trained 4)', stat='probability')
    sns.histplot(overlap_ablated, ax=ax, label='Ablated (4, trained 4)', stat='probability')
    sns.histplot(overlap_ablated_big, ax=ax, label='Big Ablated (4, trained 1-4)', stat='probability')
    ax.set_title('Ablated network')
    ax.legend()

    ax = axes[0,1]
    sns.histplot(overlap_interpolation_ref, ax=ax, label='Reference (1-4, trained 1-4)', stat='probability')
    sns.histplot(overlap_interpolation, ax=ax, label='Interpolation (1-4, trained 4)', stat='probability')
    ax.set_title('Interpolation using full network')
    ax.legend()

    ax = axes[1,1]
    sns.histplot(overlap_interpolation_ref, ax=ax, label='Reference (1-4, trained 1-4)', stat='probability') 
    sns.histplot(overlap_interpolation_ablated, ax=ax, label='Ablated (1-4, trained 4)', stat='probability')
    sns.histplot(overlap_interpolation_ablated_big, ax=ax, label='Big Ablated (1-4, trained 1-4)', stat='probability')
    ax.set_title('Interpolation using ablated network')
    ax.legend()

    ax = axes[0,2]
    sns.histplot(overlap_extrapolation_ref, ax=ax, label='Reference (4, trained 4)', stat='probability')
    sns.histplot(overlap_extrapolation, ax=ax, label='Extrapolation (5-6, trained 4)', stat='probability')
    ax.set_title('Extrapolation using full network')
    ax.legend()

    ax = axes[1,2]
    sns.histplot(overlap_extrapolation_ref, ax=ax, label='Reference (4, trained 4)', stat='probability')
    sns.histplot(overlap_extrapolation_ablated, ax=ax, label='Ablated (5-6, trained 4)', stat='probability')
    sns.histplot(overlap_extrapolation_ablated_big, ax=ax, label='Big Ablated (5-6, trained 1-4)', stat='probability')
    ax.set_title('Extrapolation using ablated network')
    ax.legend()

    fig.suptitle('Generalization summary')
    fig.tight_layout()
    fig.savefig(ROOT_OUTPUT_FOLDER + 'results/generalization_summary.pdf')
    plt.close(fig)



    # for var in ['errors_homing', 'tprs', 'reciprocal_overlaps']:
    #     for testing_condition in ['only_four', 'four_or_less', 'five_to_six']:
    #         # for net1, net2 in product(['only_four', 'ablated_four'], [['only_four', 'ablated_four']]):
    #         for net1, net2 in product(['only_four', 'four_or_less'], ['only_four', 'four_or_less']):
    #             if net1 == net2:
    #                 continue
    #             savepath = ROOT_OUTPUT_FOLDER + f'comparisons/{var}_{testing_condition}__agent_{net1}_vs_{net2}/'
    #             load_from_template = ROOT_OUTPUT_FOLDER + 'results/{}' + '__cond__' + testing_condition + '/{}.npy'
    #             d0 = np.load(load_from_template.format(net1, var))
    #             d1 = np.load(load_from_template.format(net2, var))
    #             # Note that this test is not totally symmetrical, so it's ok to call it with both orderings
    #             compare_distributions(d0, d1, [net1, net2], savepath)

if __name__ == '__main__':
    do_comparisons()