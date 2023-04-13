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

ROOT_OUTPUT_FOLDER = wdir + 'generalization_n_lines/'   
# ROOT_OUTPUT_FOLDER = wdir + 'generalization_n_lines_no_ambiguous_orderings/'


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
    for var in ['errors_homing', 'tprs', 'reciprocal_overlaps']:
        for testing_condition in ['only_four', 'four_or_less', 'five_to_six']:
            # for net1, net2 in product(['only_four', 'ablated_four'], [['only_four', 'ablated_four']]):
            for net1, net2 in product(['only_four', 'four_or_less'], ['only_four', 'four_or_less']):
                if net1 == net2:
                    continue
                savepath = ROOT_OUTPUT_FOLDER + f'comparisons/{var}_{testing_condition}__agent_{net1}_vs_{net2}/'
                load_from_template = ROOT_OUTPUT_FOLDER + 'results/{}' + '__cond__' + testing_condition + '/{}.npy'
                d0 = np.load(load_from_template.format(net1, var))
                d1 = np.load(load_from_template.format(net2, var))
                # Note that this test is not totally symmetrical, so it's ok to call it with both orderings
                compare_distributions(d0, d1, [net1, net2], savepath)

if __name__ == '__main__':
    do_comparisons()