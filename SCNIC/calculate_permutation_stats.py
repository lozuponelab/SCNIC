from statsmodels.sandbox.stats.multicomp import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm
from os.path import join
from collections import defaultdict

from SCNIC.annotate_correls import get_modules_across_rs, get_modules_to_keep
from SCNIC.calculate_permutations import filter_correls


def p_adjust(pvalues, method='fdr_bh'):
    res = multipletests(pvalues, method=method)
    return np.array(res[1], dtype=float)


def get_perms(perms_loc):
    frame_list = list()
    for path in glob(perms_loc):
        frame = pd.read_csv(path, sep='\t', index_col=(0, 1), header=None)
        frame_list.append(frame)
    combined_frames = pd.concat(frame_list, axis=1)
    return combined_frames


def perm_mannwhitneyu(x, y, dist, alternative):
    stat, _ = mannwhitneyu(x, y, alternative=alternative)
    pvalue = np.sum(stat > dist) / len(dist)
    return stat, pvalue


def perm_ttest_ind(x, y, dist, alternative='two_sided'):
    stat, _ = ttest_ind(x, y)
    if alternative == 'two-sided':
        pvalue = np.sum(np.abs(stat) > np.abs(dist)) / len(dist)
    elif alternative == 'greater':
        pvalue = np.sum(stat < dist) / len(dist)
    elif alternative == 'less':
        pvalue = np.sum(stat > dist) / len(dist)
    else:
        raise ValueError('value for alternative must be one of two-sided, greater or less')
    return stat, pvalue


def get_stats(correls, modules_across_rs, pd_perms, pd_ko_perms=None):
    stats_dfs = list()

    for min_r in tqdm(modules_across_rs.keys()):
        # going through the modules
        stats_df_index = list()
        stats_df_data = list()
        # pd ko set up
        non_cor = correls.loc[correls['correlated_%s' % min_r] == False]
        r_module_grouped = correls.groupby('module_%s' % min_r)
        for module, otus in tqdm(modules_across_rs[min_r].items()):
            if len(otus) >= 3:
                frame = r_module_grouped.get_group(module)
                # pd stats
                pd_stat, pd_pvalue = perm_ttest_ind(frame.PD, non_cor.PD, pd_perms.loc[min_r, len(otus)],
                                                    alternative='less')
                if pd_ko_perms is not None:
                    # pd ko stats
                    pd_ko_stat, pd_ko_pvalue = perm_ttest_ind(frame['residual_%s' % min_r],
                                                              non_cor['residual_%s' % min_r],
                                                              pd_ko_perms.loc[min_r, len(otus)], alternative='greater')
                else:
                    pd_ko_stat = None
                    pd_ko_pvalue = None
                # add to lists
                stats_df_index.append('%s_%s' % (min_r, module))
                stats_df_data.append((pd_stat, pd_pvalue, pd_ko_stat, pd_ko_pvalue, min_r))
        stats_df = pd.DataFrame(stats_df_data, index=stats_df_index,
                                columns=('pd_statistic', 'pd_pvalue', 'pd_ko_statistic', 'pd_ko_pvalue', 'r_level'))
        if len(stats_df) > 0:
            stats_df['pd_adj_pvalue'] = p_adjust(stats_df.pd_pvalue)
            if pd_ko_perms is not None:
                stats_df['pd_ko_adj_pvalue'] = p_adjust(stats_df.pd_ko_pvalue)
            stats_dfs.append(stats_df)
    stats_df = pd.concat(stats_dfs)
    stats_df['pd_adj_pvalue_all'] = p_adjust(stats_df.pd_pvalue)
    if pd_ko_perms is not None:
        stats_df['pd_ko_adj_pvalue_all'] = p_adjust(stats_df.pd_ko_pvalue)
    print('\n')
    return stats_df


def tabulate_stats(stats, modules_across_rs, alphas=(.01, .05, .1, .15, .2)):
    module_count = list()
    r_values = list()
    for group, frame in stats.groupby('r_level'):
        modules_greater_3 = len([module for module, otus in modules_across_rs[group].items() if len(otus) >= 3])
        if modules_greater_3 != 0:
            r_values.append(group)
            module_count.append(modules_greater_3)
    tab_stats = pd.DataFrame([module_count], columns=r_values,
                             index=['module_count'])
    tab_stats = tab_stats.transpose()
    for p_val in alphas:
        pd_sigs = list()
        pd_ko_sigs = list()
        for min_r in tab_stats.index:
            stats_min_r = stats.loc[stats.r_level == min_r]
            pd_sigs.append(np.sum(stats_min_r.pd_adj_pvalue <= p_val))
            pd_ko_sigs.append(np.sum(stats_min_r.pd_ko_adj_pvalue <= p_val))
        tab_stats['pd_sig_%s' % p_val] = pd_sigs
        tab_stats['pd_percent_sig_%s' % p_val] = pd_sigs / tab_stats.module_count
        tab_stats['pd_ko_sig_%s' % p_val] = pd_ko_sigs
        tab_stats['pd_ko_percent_sig_%s' % p_val] = pd_ko_sigs / tab_stats.module_count
    med_pd_pvalue = list()
    med_pd_ko_pvalue = list()
    for min_r in tab_stats.index:
        stats_min_r = stats.loc[stats.r_level == min_r]
        med_pd_pvalue.append(np.median(stats_min_r.pd_adj_pvalue))
        med_pd_ko_pvalue.append(np.median(stats_min_r.pd_ko_adj_pvalue))
    tab_stats['med_pd_pvalue'] = med_pd_pvalue
    tab_stats['med_pd_ko_pvalue'] = med_pd_ko_pvalue
    params_dict = defaultdict(list)
    for min_r in tab_stats.index:
        params = min_r.split('_')
        for i in range(0, len(params), 2):
            params_dict[params[i]].append(params[i + 1])
    for param, values in params_dict.items():
        tab_stats[param] = values
    return tab_stats


def make_plots(stats, tab_stats, output_loc, alphas=(.01, .05, .1, .15, .2)):
    for alpha in alphas:
        # pd_plot
        _ = sns.regplot(x='minr', y='pd_percent_sig_%s' % alpha, data=tab_stats, fit_reg=False,
                        scatter_kws={'s': tab_stats.module_count})
        plt.savefig(join(output_loc, 'pd_sig_plot_%s.png' % alpha))
        plt.clf()
        # pd_ko_plot
        _ = sns.regplot(x='minr', y='pd_ko_percent_sig_%s' % alpha, data=tab_stats, fit_reg=False,
                        scatter_kws={'s': tab_stats.module_count})
        plt.savefig(join(output_loc, 'pd_ko_sig_plot_%s.png' % alpha))
        plt.clf()
    # pvalue boxplot pd ko
    fig, ax = plt.subplots(figsize=[13, 3])
    _ = sns.boxplot(x='r_level', y='pd_ko_adj_pvalue', data=stats, ax=ax)
    plt.savefig(join(output_loc, 'pd_ko_pvalue_boxplots.png'))
    plt.clf()
    # pvalue boxplot pd
    fig, ax = plt.subplots(figsize=[13, 3])
    _ = sns.boxplot(x='r_level', y='pd_adj_pvalue', data=stats, ax=ax)
    plt.savefig(join(output_loc, 'pd_pvalue_boxplots.png'))
    plt.clf()


def do_stats(correls_loc, modules_directory_loc, perms_loc, output_loc, skip_kos=False, to_keep_loc=None,
             alphas=(.01, .05, .1, .15, .2)):
    correls = pd.read_csv(correls_loc, sep='\t', index_col=(0, 1))
    correls.index = pd.MultiIndex.from_tuples([(str(i), str(j)) for i, j in correls.index])
    if to_keep_loc is not None:
        modules_to_keep = get_modules_to_keep(to_keep_loc)
        correls = filter_correls(correls, modules_to_keep)
    else:
        modules_to_keep = None
    print('correls read')
    modules_across_rs = get_modules_across_rs(modules_directory_loc, modules_to_keep)
    print("%s modules kept" % len(modules_across_rs))
    print('modules read')
    pd_perms = get_perms(join(perms_loc, 'pd_stats_dict_*.txt'))
    if skip_kos:
        pd_ko_perms = None
    else:
        pd_ko_perms = get_perms(join(perms_loc, 'pd_ko_stats_dict_*.txt'))
    print('perms read')
    stats = get_stats(correls, modules_across_rs, pd_perms, pd_ko_perms)
    stats.to_csv(join(output_loc, 'stats.txt'), sep='\t')
    tab_stats = tabulate_stats(stats, modules_across_rs, alphas)
    tab_stats.to_csv(join(output_loc, 'tab_stats.txt'), sep='\t')
    make_plots(stats, tab_stats, output_loc, alphas)
