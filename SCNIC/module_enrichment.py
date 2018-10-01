import pandas as pd
import numpy as np
from skbio import TreeNode
from biom.table import Table
from itertools import combinations
from collections import defaultdict, OrderedDict
from glob import glob
from os.path import join
from tqdm import tqdm
import uuid
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit
from multiprocessing import Pool
from functools import partial
from statsmodels.sandbox.stats.multicomp import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import os


def p_adjust(pvalues, method='fdr_bh'):
    res = multipletests(pvalues, method=method)
    return np.array(res[1], dtype=float)


def get_modules_across_rs(module_directory_loc, verbose=False):
    modules_across_rs = OrderedDict()
    for dir_ in sorted(glob(module_directory_loc)):
        min_r = float(dir_.split('/')[-1][6:])
        min_r_modules = OrderedDict()
        with open(join(dir_, 'modules.txt')) as f:
            for line in f.readlines():
                line = line.split()
                min_r_modules[line[0]] = line[1:]
        modules_across_rs[min_r] = min_r_modules
        if verbose:
            print("There are %s modules with %s features with min R %s" %
                  (len(min_r_modules), sum([len(i) for i in list(min_r_modules.values())]), min_r))
    return modules_across_rs


def get_correlation_dicts(correls, modules_across_rs):
    correlated_items = defaultdict(list)
    module_membership = defaultdict(list)
    module_three_plus = defaultdict(list)

    for otu_pair, row in tqdm(correls.iterrows(), total=len(correls)):
        # module based things
        for min_r, modules in modules_across_rs.items():
            module_member = 'None'
            module_three_plus_member = False
            correlated = row.r > min_r
            correlated_items[min_r].append(correlated)
            if correlated:
                for module_name, otus in modules.items():  # check if otu pair is from a module
                    if otu_pair[0] in otus and otu_pair[1] in otus:
                        module_member = module_name
                        if len(otus) >= 3:
                            module_three_plus_member = True
                        break
            else:  # if not correlated include for not closed and in closed include if both are closed ref
                module_three_plus_member = True
            module_membership[min_r].append(module_member)
            module_three_plus[min_r].append(module_three_plus_member)
    print('\n')
    return correlated_items, module_membership, module_three_plus


def percent_shared(otu_i_arr, otu_j_arr):
    otu_data = np.stack((otu_i_arr, otu_j_arr)).astype(bool).sum(axis=0)
    shared = np.sum(otu_data == 2)
    return shared / np.sum(otu_data > 0)


def annotate_correls(correls, correls_tip_tips, genome_table, correlated_items, module_membership, module_three_plus):
    new_index = list()
    new_rows = list()
    new_index.append('PD')
    new_rows.append([correls_tip_tips[otu_pair] for otu_pair in correls.index])
    print('pd acquired')
    new_index.append('percent_shared')
    new_rows.append([percent_shared(genome_table.data(otu_pair[0]), genome_table.data(otu_pair[1]))
                     for otu_pair in correls.index])
    print('percent shared acquired')
    for min_r, list_ in correlated_items.items():
        new_index.append('correlated_%s' % min_r)
        new_rows.append(list_)
    for min_r, list_ in module_membership.items():
        new_index.append('module_%s' % min_r)
        new_rows.append(list_)
    for min_r, list_ in module_three_plus.items():
        new_index.append('three_plus_%s' % min_r)
        new_rows.append(list_)
    print('correlation stats acquired')
    new_df = pd.DataFrame(new_rows, columns=correls.index, index=new_index)
    return pd.merge(correls, new_df.transpose(), left_index=True, right_index=True)


def do_annotate_correls(correls_loc, tre_loc, genome_loc, module_loc, output_loc):
    correls = pd.read_table(correls_loc, index_col=(0, 1))
    correls.index = pd.MultiIndex.from_tuples([(str(i), str(j)) for i, j in correls.index])
    print("read correls")
    tre = TreeNode.read(tre_loc)
    correls_tip_tips = tre.tip_tip_distances(set([otu for otu_pair in correls.index for otu in otu_pair]))
    print("read tree")
    genome_frame = pd.read_table(genome_loc, index_col=0)
    genome_frame = genome_frame.loc[set([otu for otu_pair in correls.index for otu in otu_pair])]
    genome_frame = genome_frame.loc[:, genome_frame.sum(axis=0) > 0]
    genome_table = Table(genome_frame.transpose().values, observation_ids=genome_frame.columns,
                         sample_ids=genome_frame.index)
    print("read table")
    modules_across_rs = get_modules_across_rs(module_loc)
    correlated_items, module_membership, module_three_plus = get_correlation_dicts(correls, modules_across_rs)
    correls = annotate_correls(correls, correls_tip_tips, genome_table, correlated_items, module_membership,
                               module_three_plus)
    print("annotated correls")
    correls.to_csv(output_loc, sep='\t')


############################


def func(x, a, b, c, psuedocount=10e-20):
    return a*np.log(x+b+psuedocount)+c


def get_module_sizes_across_rs(modules_across_rs):
    module_sizes_across_rs = dict()
    for min_r, modules in modules_across_rs.items():
        module_sizes = list()
        for module, otus in modules.items():
            module_sizes.append(len(otus))
        module_sizes_across_rs[min_r] = set(module_sizes)
    return module_sizes_across_rs


def calc_residuals(actual_x, actual_y, popt):
    return np.array(actual_y, dtype=np.float64) - func(np.array(actual_x, dtype=np.float64), *popt)


def calc_popt(x, y):
    popt, _ = curve_fit(func, np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))
    return popt


def perm(random_module_otus, correls, non_cor_pd, non_cor_residuals, popt_non_cor):
    pairs = list()
    for otu_i, otu_j in combinations(random_module_otus, 2):
        if (otu_i, otu_j) in correls.index:
            pairs.append((otu_i, otu_j))
        else:
            pairs.append((otu_j, otu_i))
    random_module_correls = correls.loc[pairs]
    # pd stuff
    pd_res, _ = mannwhitneyu(random_module_correls.PD, non_cor_pd, alternative='two-sided')
    # pd ko stuff
    residuals = calc_residuals(random_module_correls.PD, random_module_correls.percent_shared, popt_non_cor)
    pd_ko_res, _ = mannwhitneyu(residuals, non_cor_residuals, alternative='greater')
    return pd_res, pd_ko_res


def run_perms(correls, perms, procs, module_sizes, output_loc):
    min_rs = sorted([float(i.split('_')[-1]) for i in correls.columns if 'module_' in i])
    current_milli_time = uuid.uuid4()
    all_otus = tuple(set([otu for pair in correls.index for otu in pair]))
    os.makedirs(output_loc, exist_ok=True)
    for min_r in tqdm(min_rs):
        # pd ko set up
        non_cor = correls.loc[correls['correlated_%s' % min_r] == False]
        popt_non_cor = calc_popt(non_cor.PD, non_cor.percent_shared)
        non_cor_residuals = calc_residuals(non_cor.PD, non_cor.percent_shared, popt_non_cor)
        # perms
        pd_stats_dict = dict()
        pd_ko_stats_dict = dict()
        for size in tqdm(module_sizes[min_r]):
            if size < 3:
                continue
            pool = Pool(processes=procs)
            partial_func = partial(perm, correls=correls, non_cor_pd=non_cor.PD, non_cor_residuals=non_cor_residuals,
                                   popt_non_cor=popt_non_cor)
            results = pool.map(partial_func, (np.random.choice(all_otus, size, replace=False) for i in range(perms)))
            pool.close()
            pool.join()
            pd_stats_dict[size] = np.array([i[0] for i in results])
            pd_ko_stats_dict[size] = np.array([i[1] for i in results])

        # print dict to file
        with open(join(output_loc, 'pd_stats_dict_%s.txt' % current_milli_time), 'a') as f:
            for key, values in pd_stats_dict.items():
                f.write('%s\t%s\t%s\n' % (min_r, key, '\t'.join([str(i) for i in values])))
        with open(join(output_loc, 'pd_ko_stats_dict_%s.txt' % current_milli_time), 'a') as f:
            for key, values in pd_ko_stats_dict.items():
                f.write('%s\t%s\t%s\n' % (min_r, key, '\t'.join([str(i) for i in values])))
    print('\n')


def do_multiprocessed_perms(correls_loc, perms, procs, modules_directory_loc, output_loc):
    modules_across_rs = get_modules_across_rs(modules_directory_loc)
    module_sizes_across_rs = get_module_sizes_across_rs(modules_across_rs)
    print("got module sizes")
    correls = pd.read_table(correls_loc, index_col=(0, 1))
    correls.index = pd.MultiIndex.from_tuples([(str(i), str(j)) for i, j in correls.index])
    print("read correls")
    run_perms(correls, perms, procs, module_sizes_across_rs, output_loc)


############################


def get_perms(perms_loc):
    frame_list = list()
    for path in glob(perms_loc):
        frame = pd.read_table(path, index_col=(0, 1), header=None)
        frame_list.append(frame)
    combined_frames = pd.concat(frame_list, axis=1)
    return combined_frames


def perm_mannwhitneyu(x, y, dist, alternative):
    stat, _ = mannwhitneyu(x, y, alternative=alternative)
    pvalue = np.sum(stat < dist) / len(dist)
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
        popt_non_cor = calc_popt(non_cor.PD, non_cor.percent_shared)
        non_cor_residuals = calc_residuals(non_cor.PD, non_cor.percent_shared, popt_non_cor)
        for module, otus in tqdm(modules_across_rs[min_r].items()):
            if len(otus) >= 3:
                frame = r_module_grouped.get_group(module)
                # pd stats
                pd_stat, pd_pvalue = perm_mannwhitneyu(frame.PD, non_cor.PD, pd_perms.loc[min_r, len(otus)],
                                                       alternative='less')
                # pd ko stats
                residuals = calc_residuals(frame.PD, frame.percent_shared, popt_non_cor)
                pd_ko_stat, pd_ko_pvalue = perm_mannwhitneyu(residuals, non_cor_residuals,
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
    _ = sns.regplot(x='index', y='pd_percent_sig', data=tab_stats.reset_index(), fit_reg=False,
                    scatter_kws={'s': tab_stats.module_count})
    plt.savefig(join(output_loc, 'pd_sig_plot.png'))
    plt.clf()
    _ = sns.regplot(x='index', y='pd_ko_percent_sig', data=tab_stats.reset_index(), fit_reg=False,
                    scatter_kws={'s': tab_stats.module_count})
    plt.savefig(join(output_loc, 'pd_ko_sig_plot.png'))
