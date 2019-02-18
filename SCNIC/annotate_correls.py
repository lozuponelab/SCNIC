import pandas as pd
import numpy as np
from skbio import TreeNode
from biom.table import Table
from collections import defaultdict, OrderedDict
from glob import glob
from tqdm import tqdm
from scipy.optimize import curve_fit


def log_linear_func(x, a, b, c, psuedocount=10e-20):
    return a*np.log(x+b+psuedocount)+c


def genome_frame_to_table(genome_frame, otus_to_keep):
    if otus_to_keep is not None:
        genome_frame = genome_frame.loc[otus_to_keep]
    genome_frame = genome_frame.loc[:, genome_frame.sum(axis=0) > 0]
    genome_table = Table(genome_frame.transpose().values, observation_ids=genome_frame.columns,
                         sample_ids=genome_frame.index)
    return genome_table


def get_modules(premodules):
    """premodules is a list of strings for file like object where each line is a module, first value is module name
    and subsequent values are observations in that module. All are separated by tabs.
    """
    modules = OrderedDict()
    for line in premodules:
        line = line.strip().split('\t')
        modules[line[0]] = line[1:]
    return modules


def get_modules_to_keep(folders_to_keep_loc):
    with open(folders_to_keep_loc) as f:
        folders_to_keep = list()
        for line in f:
            folders_to_keep.append(line.strip())
        return folders_to_keep


def get_modules_across_rs(modules_loc, modules_to_keep=None, verbose=False):
    modules_across_rs = OrderedDict()
    for module_loc in sorted(glob(modules_loc)):
        with open(module_loc) as f:
            key = module_loc.split('/')[-2]
            modules_across_rs[key] = get_modules(f.readlines())
        if verbose:
            print("There are %s modules with %s features with min R %s" %
                  (len(modules_across_rs[key]),
                   sum([len(i) for i in list(modules_across_rs[key])]), key))
    if modules_to_keep is not None:
        modules_across_rs = {key: value for key, value in modules_across_rs.items() if key in modules_to_keep}
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
            params = min_r.split('_')
            param_dict = {params[i]: float(params[i+1]) for i in range(0, len(params), 2)}
            correlated = row.r >= param_dict['minr']
            correlated_items[min_r].append(correlated)
            for module_name, otus in modules.items():  # check if otu pair is from a module
                if otu_pair[0] in otus and otu_pair[1] in otus:
                    module_member = module_name
                    if len(otus) >= 3:
                        module_three_plus_member = True
                    break
            if module_member == 'None':
                module_three_plus_member = True
            module_membership[min_r].append(module_member)
            module_three_plus[min_r].append(module_three_plus_member)
    print('\n')
    return correlated_items, module_membership, module_three_plus


def add_correlation_dicts(correls, correlated_items, module_membership, module_three_plus):
    new_index = list()
    new_rows = list()
    for min_r, list_ in correlated_items.items():
        new_index.append('correlated_%s' % min_r)
        new_rows.append(list_)
    for min_r, list_ in module_membership.items():
        new_index.append('module_%s' % min_r)
        new_rows.append(list_)
    for min_r, list_ in module_three_plus.items():
        new_index.append('three_plus_%s' % min_r)
        new_rows.append(list_)
    new_df = pd.DataFrame(new_rows, columns=correls.index, index=new_index)
    return new_df.transpose()


def percent_shared(otu_i_arr, otu_j_arr):
    otu_data = np.stack((otu_i_arr, otu_j_arr)).astype(bool).sum(axis=0)
    shared = np.sum(otu_data == 2)
    return shared / np.sum(otu_data > 0)


def add_pd_ko_data(correls, correls_tip_tips, genome_table):
    new_index = list()
    new_rows = list()
    new_index.append('PD')
    new_rows.append([correls_tip_tips[otu_pair] for otu_pair in correls.index])
    new_index.append('percent_shared')
    new_rows.append([percent_shared(genome_table.data(otu_pair[0]), genome_table.data(otu_pair[1]))
                     for otu_pair in correls.index])
    new_df = pd.DataFrame(new_rows, columns=correls.index, index=new_index)
    return new_df.transpose()


def calc_popt(x, y, func):
    popt, _ = curve_fit(func, np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))
    return popt


def calc_residuals(actual_x, actual_y, popt, func):
    return np.array(actual_y, dtype=np.float64) - func(np.array(actual_x, dtype=np.float64), *popt)


def get_residuals_across_rs(correlation_data, pd_ko_df, modules_across_rs, func):
    popt_across_rs = dict()
    for i, min_r in enumerate(modules_across_rs.keys()):
        noncor_correls = pd_ko_df[~correlation_data['correlated_%s' % min_r]]
        try:
            popt = calc_popt(noncor_correls.PD, noncor_correls.percent_shared, func)
            popt_across_rs[min_r] = popt
        except RuntimeError:
            raise RuntimeError('curve fit broke: %s, iter %s' % (min_r, i))
    new_df_columns = ['residual_%s' % min_r for min_r in modules_across_rs.keys()]
    new_df_data = list()
    for otu_pair, row in pd_ko_df.iterrows():
        new_row = [calc_residuals(row.PD, row.percent_shared, popt_across_rs[min_r], func)
                   for min_r in modules_across_rs.keys()]
        new_df_data.append(new_row)
    new_df = pd.DataFrame(new_df_data, index=correlation_data.index, columns=new_df_columns)
    return new_df


def do_annotate_correls(correls_loc, tre_loc, genome_loc, module_loc, output_loc, skip_kos=False, modules_to_keep_loc=None,
                        func=log_linear_func):
    correls = pd.read_csv(correls_loc, sep='\t', index_col=(0, 1))
    correls.index = pd.MultiIndex.from_tuples([(str(i), str(j)) for i, j in correls.index])
    print("read correls")
    tre = TreeNode.read(tre_loc)
    correls_tip_tips = tre.tip_tip_distances(set([otu for otu_pair in correls.index for otu in otu_pair]))
    print("read tree")
    genome_frame = pd.read_csv(genome_loc, sep='\t', index_col=0)
    genome_table = genome_frame_to_table(genome_frame, set([otu for otu_pair in correls.index for otu in otu_pair]))
    print("read table")
    if modules_to_keep_loc is not None:
        modules_to_keep = get_modules_to_keep(modules_to_keep_loc)
    else:
        modules_to_keep = None
    modules_across_rs = get_modules_across_rs(module_loc, modules_to_keep=modules_to_keep)
    print("read modules")
    correlated_items, module_membership, module_three_plus = get_correlation_dicts(correls, modules_across_rs)
    correlation_df = add_correlation_dicts(correls, correlated_items, module_membership,
                                           module_three_plus)
    print("added correlation data")
    pd_ko_df = add_pd_ko_data(correls, correls_tip_tips, genome_table)
    print("added pd ko data")
    if not skip_kos:
        residual_df = get_residuals_across_rs(correlation_df, pd_ko_df, modules_across_rs, func)
        print("added residuals")
        correls = pd.concat([correls, pd_ko_df, residual_df, correlation_df], axis=1)
    else:
        correls = pd.concat([correls, pd_ko_df, correlation_df], axis=1)
    correls.to_csv(output_loc, sep='\t')
