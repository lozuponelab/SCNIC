from statsmodels.sandbox.stats.multicomp import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm
from os.path import join

from SCNIC.annotate_correls import get_modules_across_rs


def p_adjust(pvalues, method='fdr_bh'):
    res = multipletests(pvalues, method=method)
    return np.array(res[1], dtype=float)


def get_perms(perms_loc):
    frame_list = list()
    for path in glob(perms_loc):
        frame = pd.read_table(path, index_col=(0, 1), header=None)
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


def get_stats(correls, modules_across_rs, pd_perms, pd_ko_perms):
    stats_dfs = list()
    min_rs = sorted([float(i.split('_')[-1]) for i in correls.columns if 'module_' in i])

    for min_r in tqdm(min_rs):
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
                # pd ko stats
                pd_ko_stat, pd_ko_pvalue = perm_ttest_ind(frame['residual_%s' % min_r], non_cor['residual_%s' % min_r],
                                           pd_ko_perms.loc[min_r, len(otus)], alternative='greater')
                # add to lists
                stats_df_index.append('%s_%s' % (min_r, module))
                stats_df_data.append((pd_stat, pd_pvalue, pd_ko_stat, pd_ko_pvalue, min_r))
        stats_df = pd.DataFrame(stats_df_data, index=stats_df_index,
                                columns=('pd_statistic', 'pd_pvalue', 'pd_ko_statistic', 'pd_ko_pvalue', 'r_level'))
        if len(stats_df) > 0:
            stats_df['pd_adj_pvalue'] = p_adjust(stats_df.pd_pvalue)
            stats_df['pd_ko_adj_pvalue'] = p_adjust(stats_df.pd_ko_pvalue)
            stats_dfs.append(stats_df)
    stats_df = pd.concat(stats_dfs)
    print('\n')
    return stats_df


def tabulate_stats(stats, modules_across_rs, alpha=.05):
    module_count = list()
    pd_ko_sig = list()
    pd_ko_percent_sig = list()
    pd_sig = list()
    pd_percent_sig = list()
    r_values = list()
    for group, frame in stats.groupby('r_level'):
        modules_greater_3 = len([module for module, otus in modules_across_rs[group].items() if len(otus) >= 3])
        if modules_greater_3 != 0:
            r_values.append(group)
            module_count.append(modules_greater_3)
            pd_sig_frame = frame.loc[frame.pd_adj_pvalue < alpha]
            pd_sig.append(pd_sig_frame.shape[0])
            pd_percent_sig.append(pd_sig_frame.shape[0]/modules_greater_3)
            pd_ko_sig_frame = frame.loc[frame.pd_ko_adj_pvalue < alpha]
            pd_ko_sig.append(pd_ko_sig_frame.shape[0])
            pd_ko_percent_sig.append(pd_ko_sig_frame.shape[0]/modules_greater_3)
    tab_stats = pd.DataFrame([module_count, pd_sig, pd_percent_sig, pd_ko_sig, pd_ko_percent_sig], columns=r_values,
                             index=('module_count', 'pd_sig', 'pd_percent_sig', 'pd_ko_sig', 'pd_ko_percent_sig'))
    return tab_stats.transpose()


def make_plots(stats, tab_stats, output_loc):
    # pd_plot
    _ = sns.regplot(x='index', y='pd_percent_sig', data=tab_stats.reset_index(), fit_reg=False,
                    scatter_kws={'s': tab_stats.module_count})
    plt.savefig(join(output_loc, 'pd_sig_plot.png'))
    plt.clf()
    # pd_ko_plot
    _ = sns.regplot(x='index', y='pd_ko_percent_sig', data=tab_stats.reset_index(), fit_reg=False,
                    scatter_kws={'s': tab_stats.module_count})
    plt.savefig(join(output_loc, 'pd_ko_sig_plot.png'))
    plt.clf()
    # pvalue boxplot pd ko
    fig, ax = plt.subplots(figsize=[13, 3])
    _ = sns.boxplot(x='r_level', y='pd_ko_adj_pvalue', data=stats, ax=ax)
    plt.savefig(join(output_loc, 'pd_ko_pvalue_boxplots.png'))
    plt.clf()
    #pvalue boxplot pd
    fig, ax = plt.subplots(figsize=[13, 3])
    _ = sns.boxplot(x='r_level', y='pd_adj_pvalue', data=stats, ax=ax)
    plt.savefig(join(output_loc, 'pd_pvalue_boxplots.png'))
    plt.clf()


def do_stats(correls_loc, modules_directory_loc, perms_loc, output_loc, alpha=.05):
    correls = pd.read_table(correls_loc, index_col=(0, 1))
    correls.index = pd.MultiIndex.from_tuples([(str(i), str(j)) for i, j in correls.index])
    print('correls read')
    modules_across_rs = get_modules_across_rs(modules_directory_loc)
    print('modules read')
    pd_perms = get_perms(join(perms_loc, 'pd_stats_dict_*.txt'))
    pd_ko_perms = get_perms(join(perms_loc, 'pd_ko_stats_dict_*.txt'))
    print('perms read')
    stats = get_stats(correls, modules_across_rs, pd_perms, pd_ko_perms)
    stats.to_csv(join(output_loc, 'stats.txt'), sep='\t')
    tab_stats = tabulate_stats(stats, modules_across_rs, alpha)
    tab_stats.to_csv(join(output_loc, 'tab_stats.txt'), sep='\t')
    make_plots(stats, tab_stats, output_loc)
