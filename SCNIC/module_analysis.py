from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import squareform
from skbio.tree import TreeNode
from itertools import combinations
import pandas as pd
import numpy as np
from biom.table import Table
from biom.util import biom_open
import os


def correls_to_cor(correls, metric='r'):
    ids = sorted(set([j for i in correls.index for j in i]))
    data = np.zeros((len(ids), len(ids)))
    for i in range(len(ids)):
        for j in range(i, len(ids)):
            if i == j:
                data[i, i] = 1
            else:
                id_i = ids[i]
                id_j = ids[j]
                try:
                    cor = correls.loc[(id_i, id_j), metric]
                except KeyError:
                    cor = correls.loc[(id_j, id_i), metric]
                data[i, j] = cor
                data[j, i] = cor
    return squareform(data, checks=False), ids


def cor_to_dist(cor):
    # convert from correlation to distance
    return 1 - ((cor + 1) / 2)


def make_modules(dist, min_dist, obs_ids, prefix="module"):
    # create linkage matrix using complete linkage
    z = complete(dist)
    # make tree from linkage matrix with names from dist
    tree = TreeNode.from_linkage_matrix(z, obs_ids)
    # get all tips so in the end we can check if we are done
    all_tips = len([i for i in tree.postorder() if i.is_tip()])
    modules = set()
    seen = set()
    dist = pd.DataFrame(squareform(dist), index=obs_ids, columns=obs_ids)
    for node in tree.levelorder():
        if node.is_tip():
            seen.add(node.name)
        else:
            tip_names = frozenset((i.name for i in node.postorder() if i.is_tip()))
            if tip_names.issubset(seen):
                continue
            dists = (dist.loc[tip1, tip2] > min_dist for tip1, tip2 in combinations(tip_names, 2))
            if any(dists):
                continue
            else:
                modules.add(tip_names)
                seen.update(tip_names)
        if len(seen) == all_tips:
            modules = {'%s_%s' % (prefix, i): otus for i, otus in enumerate(sorted(modules, key=len, reverse=True))}
            return modules
    raise ValueError("Well, how did I get here?")


def collapse_modules(table, modules):
    """collapse created modules in a biom table, members of multiple modules will be added to the smallest module"""
    table = table.copy()
    module_array = np.zeros((len(modules), table.shape[1]))

    seen = set()
    for module_, otus in modules.items():
        module_number = int(module_.split('_')[-1])
        seen = seen | otus
        # sum everything in the module
        module_array[module_number] = np.sum([table.data(feature, axis="observation") for feature in otus], axis=0)

    table.filter(seen, axis='observation', invert=True)

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), module_array))
    new_table_obs = list(table.ids(axis='observation')) + list(modules.keys())
    return Table(new_table_matrix, new_table_obs, table.ids())


def write_modules_to_dir(table, modules):
    # for each module merge values and print modules to file
    os.makedirs("modules")
    for module_, otus in modules.items():
        # make biom tables for each module and write to file
        module_table = table.filter(otus, axis='observation', inplace=False)
        with biom_open("modules/%s.biom" % module_, 'w') as f:
            module_table.to_hdf5(f, 'SCNIC.module_analysis.write_modules_to_dir')


def write_modules_to_file(modules, prefix="module", path_str='modules.txt'):
    # write all modules to file
    with open(path_str, 'w') as f:
        for module_, otus in modules.items():
            f.write('%s\t%s\n' % (module_, '\t'.join([str(otu) for otu in otus])))


def add_modules_to_metadata(modules, metadata):
    """
    modules is dict of otus, metadata is a dictionary of dictionaries where outer dict keys
    are features, inner dict keys are metadata names and values are metadata values
    """
    for module_, otus in modules.items():
        for otu in otus:
            try:
                metadata[str(otu)]['module'] = module_
            except KeyError:
                metadata[str(otu)] = dict()
                metadata[str(otu)]['module'] = module_
    return metadata
