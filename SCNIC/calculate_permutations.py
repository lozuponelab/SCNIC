"""
Generate permutation datasets for SCNIC statistical analysis.

This module provides functions for sampling random modules, computing
permutation statistics, and writing results to disk.
"""

from itertools import combinations
import uuid
from scipy.stats import ttest_ind
import multiprocessing
from functools import partial
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from os.path import join

from SCNIC.annotate_correls import get_modules_across_rs


def get_module_sizes_across_rs(modules_across_rs):
    """
    Compute module size sets for each correlation threshold.

    Parameters
    ----------
    modules_across_rs : dict
        Mapping of threshold keys to module dictionaries.

    Returns
    -------
    dict
        Mapping of thresholds to unique module sizes.
    """
    module_sizes_across_rs = dict()
    for min_r, modules in modules_across_rs.items():
        module_sizes = list()
        for module, otus in modules.items():
            module_sizes.append(len(otus))
        module_sizes_across_rs[min_r] = set(module_sizes)
    return module_sizes_across_rs


def get_modules_to_keep(folders_to_keep_loc):
    """
    Read module names to keep from a file.

    Parameters
    ----------
    folders_to_keep_loc : str
        Path to a newline-delimited file listing module names.

    Returns
    -------
    list
        Module names to keep.
    """
    with open(folders_to_keep_loc) as f:
        folders_to_keep = list()
        for line in f:
            folders_to_keep.append(line.strip())
        return folders_to_keep


def perm(random_module_otus, correls, min_r, skip_ko=False):
    """
    Compute permutation statistics for one sampled module.

    Parameters
    ----------
    random_module_otus : sequence
        Randomly sampled OTUs for the module.
    correls : pandas.DataFrame
        Correlation dataframe indexed by OTU pairs.
    min_r : float
        Correlation threshold used for defining non-correlated pairs.
    skip_ko : bool, default False
        If True, return only PD statistics.

    Returns
    -------
    tuple or float
        PD statistic or (PD, PD KO) statistics.
    """
    pairs = list()
    for pair in combinations(random_module_otus, 2):
        pairs.append(tuple(sorted(pair)))
    random_module_correls = correls.loc[pairs]
    non_cor_correls = correls.loc[~correls['correlated_%s' % min_r]]
    # pd stuff
    pd_res, _ = ttest_ind(random_module_correls.PD, non_cor_correls.PD)
    if skip_ko:
        return pd_res
    else:
        # pd ko stuff
        pd_ko_res, _ = ttest_ind(random_module_correls['residual_%s' % min_r], non_cor_correls['residual_%s' % min_r])
        return pd_res, pd_ko_res


def filter_correls(correls, to_keep):
    """
    Filter correlation columns to only those matching requested module parameters.

    Parameters
    ----------
    correls : pandas.DataFrame
        Input correlation statistics.
    to_keep : iterable
        Parameter strings to keep.

    Returns
    -------
    pandas.DataFrame
        Filtered correlations.
    """
    cols_to_keep = list(correls.columns[:3])
    for column in correls.columns[3:]:
        if 'gamma' in column:
            params = '_'.join(column.split('_')[-4:])
        else:
            params = '_'.join(column.split('_')[-2:])
        if params in to_keep:
            cols_to_keep.append(column)
    return correls[cols_to_keep]


def run_perms(correls, perms, procs, module_sizes, output_loc, skip_ko=False):
    """
    Run permutation tests for each module size and write results to files.

    Parameters
    ----------
    correls : pandas.DataFrame
        Correlation dataframe indexed by OTU pairs.
    perms : int
        Number of permutations per module size.
    procs : int
        Number of worker processes.
    module_sizes : dict
        Mapping of correlation thresholds to module sizes.
    output_loc : str
        Directory for output files.
    skip_ko : bool, default False
        If True, do not compute PD KO statistics.
    """
    current_milli_time = uuid.uuid4()
    all_otus = tuple(set([otu for pair in correls.index for otu in pair]))
    os.makedirs(output_loc, exist_ok=True)
    for min_r in tqdm(module_sizes.keys()):
        # perms
        correls_perm = filter_correls(correls, (min_r,))
        pd_stats_dict = dict()
        pd_ko_stats_dict = dict()
        for size in tqdm(module_sizes[min_r]):
            if size < 3:
                continue

            ## madi update: bc multiprocessing throwing deadlock/deprecation errors when nproc=1
            ## nprocs <= 1 runs sequentially so no child process is spawned
            ## nprocs > 1 now uses spawn instead of default fork - avoids python 3.12 warning when parent is already multithreaded 
            partial_func = partial(perm, correls=correls_perm, min_r=min_r, skip_ko=skip_ko)
            if procs <= 1:
                results = [partial_func(np.random.choice(all_otus, size, replace=False))
                           for _ in range(perms)]
            else:
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(processes=procs) as pool:
                    results = pool.map(partial_func,
                                       (np.random.choice(all_otus, size, replace=False)
                                        for _ in range(perms)))
            if skip_ko:
                pd_stats_dict[size] = np.array(results)
            else:
                pd_stats_dict[size] = np.array([i[0] for i in results])
                pd_ko_stats_dict[size] = np.array([i[1] for i in results])

        # print dict to file
        with open(join(output_loc, 'pd_stats_dict_%s.txt' % current_milli_time), 'a') as f:
            for key, values in pd_stats_dict.items():
                f.write('%s\t%s\t%s\n' % (min_r, key, '\t'.join([str(i) for i in values])))
        if not skip_ko:
            with open(join(output_loc, 'pd_ko_stats_dict_%s.txt' % current_milli_time), 'a') as f:
                for key, values in pd_ko_stats_dict.items():
                    f.write('%s\t%s\t%s\n' % (min_r, key, '\t'.join([str(i) for i in values])))
    print('\n')


def do_multiprocessed_perms(correls_loc, perms, procs, modules_directory_loc, output_loc, skip_kos,
                            folders_to_keep_loc=None):
    """
    Driver for the permutation generation workflow.

    Parameters
    ----------
    correls_loc : str
        Path to the input correlation table.
    perms : int
        Number of permutations to run.
    procs : int
        Number of worker processes.
    modules_directory_loc : str
        Directory or glob pattern containing modules.
    output_loc : str
        Output directory for permutation results.
    skip_kos : bool
        If True, skip permutation statistics for KO.
    folders_to_keep_loc : str or None
        Optional file with module names to keep.
    """
    if folders_to_keep_loc is not None:
        modules_to_keep = get_modules_to_keep(folders_to_keep_loc)
    else:
        modules_to_keep = None
    modules_across_rs = get_modules_across_rs(modules_directory_loc, modules_to_keep)
    print("%s modules found" % len(modules_across_rs))
    module_sizes_across_rs = get_module_sizes_across_rs(modules_across_rs)
    print("got module sizes")
    correls = pd.read_csv(correls_loc, sep='\t', index_col=(0, 1))
    correls.index = pd.MultiIndex.from_tuples([tuple(sorted((str(i), str(j)))) for i, j in correls.index])
    if folders_to_keep_loc is not None:
        correls = filter_correls(correls, modules_to_keep)
    print("read correls")
    run_perms(correls, perms, procs, module_sizes_across_rs, output_loc, skip_ko=skip_kos)
