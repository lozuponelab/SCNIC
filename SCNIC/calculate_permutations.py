from itertools import combinations
import uuid
from scipy.stats import ttest_ind
from multiprocessing import Pool
from functools import partial
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from os.path import join

from SCNIC.annotate_correls import get_modules_across_rs


def get_module_sizes_across_rs(modules_across_rs):
    module_sizes_across_rs = dict()
    for min_r, modules in modules_across_rs.items():
        module_sizes = list()
        for module, otus in modules.items():
            module_sizes.append(len(otus))
        module_sizes_across_rs[min_r] = set(module_sizes)
    return module_sizes_across_rs


def get_modules_to_keep(folders_to_keep_loc):
    with open(folders_to_keep_loc) as f:
        folders_to_keep = list()
        for line in f:
            folders_to_keep.append(line.strip())
        return folders_to_keep


def perm(random_module_otus, correls, min_r, skip_ko=False):
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
            pool = Pool(processes=procs)
            partial_func = partial(perm, correls=correls_perm, min_r=min_r, skip_ko=skip_ko)
            results = pool.map(partial_func, (np.random.choice(all_otus, size, replace=False) for _ in range(perms)))
            pool.close()
            pool.join()
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
