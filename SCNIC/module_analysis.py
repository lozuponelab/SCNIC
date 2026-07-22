"""
Algorithms for making modules from correlation networks.

This module supports naive tree-based modules, k-clique communities,
Louvain community detection, table collapsing, and metadata assignment.
"""

from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import squareform
from skbio.tree import TreeNode
from itertools import combinations
import pandas as pd
import numpy as np
from biom.table import Table
from biom.util import biom_open
import os
import networkx as nx
from community import community_louvain
from collections import defaultdict

from SCNIC import general


def correls_to_cor(correls, metric='r'):
    """
    Convert a correlation edge list to a condensed correlation matrix.

    Parameters
    ----------
    correls : pandas.DataFrame
    metric : str, default 'r'

    Returns
    -------
    tuple
        (condensed distance array, label list)
    """
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
    """
    Convert correlation coefficient to distance.

    Parameters
    ----------
    cor : float or ndarray

    Returns
    -------
    float or ndarray
    """
    return 1 - ((cor + 1) / 2)


def make_modules_naive(correls, min_r=None, max_p=None, prefix="module"):
    """
    Build modules using hierarchical clustering and a correlation threshold.

    Parameters
    ----------
    correls : pandas.DataFrame
    min_r : float or None
    max_p : float or None
    prefix : str, default 'module'

    Returns
    -------
    dict
    """
    # read in correlations file and make distance matrix
    if min_r is not None:
        min_dist = cor_to_dist(min_r)
        cor, labels = correls_to_cor(correls)
        dist = cor_to_dist(cor)
    elif max_p is not None:
        # TODO: This
        raise NotImplementedError('Making modules based on a p-value is not currently supported')
    else:
        raise ValueError("this is prevented above")
    # create linkage matrix using complete linkage
    z = complete(dist)
    # make tree from linkage matrix with names from dist
    tree = TreeNode.from_linkage_matrix(z, labels)
    # get all tips so in the end we can check if we are done
    all_tips = tree.count(tips=True)
    modules = set()
    seen = set()
    dist = pd.DataFrame(squareform(dist), index=labels, columns=labels)
    for node in tree.levelorder():
        if node.is_tip():
            seen.add(node.name)
        else:
            tip_names = frozenset((i.name for i in node.tips()))
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


def make_modules_k_cliques(correls, min_r=None, max_p=None, k=3, prefix="module"):
    """
    Build modules from a correlation network using k-clique communities.

    Parameters
    ----------
    correls : pandas.DataFrame
    min_r : float or None
    max_p : float or None
    k : int, default 3
    prefix : str, default 'module'

    Returns
    -------
    dict
    """
    correls_filt = general.filter_correls(correls, max_p=max_p, min_r=min_r, conet=True)
    net = general.correls_to_net(correls_filt)
    premodules = list(nx.algorithms.community.k_clique_communities(net, k))
    # reverse modules so observations will be added to smallest modules
    premodules.sort(key=len, reverse=True)

    modules = dict()
    seen = set()
    for i, module in enumerate(premodules):
        # process module
        module = module-seen
        seen = seen | module
        modules[prefix+"_"+str(i)] = module
        for node in module:
            net.nodes[node][prefix] = i ## networkx deprecated net.node - is now net.nodes
    return modules


def make_modules_louvain(correls, min_r=None, max_p=None, gamma=.01, prefix="module"):
    """
    Build modules using the Louvain community detection algorithm.

    Parameters
    ----------
    correls : pandas.DataFrame
    min_r : float or None
    max_p : float or None
    gamma : float, default .01
    prefix : str, default 'module'

    Returns
    -------
    dict
    """
    correls_filt = general.filter_correls(correls, max_p=max_p, min_r=min_r, conet=True)
    net = general.correls_to_net(correls_filt)
    partition = community_louvain.best_partition(net, resolution=gamma) ## was just louvain.best_partition() had to be changed since i dont think that will work

    premodules = defaultdict(list)
    for otu, module in partition.items():
        premodules[module].append(otu)
    premodules = list(premodules.values())
    premodules.sort(key=len, reverse=True)

    modules = dict()
    for i, otus in enumerate(premodules):
        if len(otus) > 1:
            modules['%s_%s' % (prefix, i)] = otus
    return modules


def collapse_modules(table, modules):
    """
    Collapse BIOM table observations into module sums.

    Parameters
    ----------
    table : biom.table.Table
    modules : dict

    Returns
    -------
    biom.table.Table
    """
    table = table.copy()
    module_array = np.zeros((len(modules), table.shape[1]))

    seen = set()
    for module_, otus in modules.items():
        module_number = int(module_.split('_')[-1])
        seen = seen | set(otus)
        # sum everything in the module
        module_array[module_number] = np.sum([table.data(feature, axis="observation") for feature in otus], axis=0)

    table.filter(seen, axis='observation', invert=True)

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), module_array))
    new_table_obs = list(table.ids(axis='observation')) + list(modules.keys())
    return Table(new_table_matrix, new_table_obs, table.ids())


def write_modules_to_dir(table, modules):
    """
    Write one BIOM file per module in a modules directory.

    Parameters
    ----------
    table : biom.table.Table
    modules : dict
    """
    os.makedirs("modules")
    for module_, otus in modules.items():
        # make biom tables for each module and write to file
        module_table = table.filter(otus, axis='observation', inplace=False)
        with biom_open("modules/%s.biom" % module_, 'w') as f:
            module_table.to_hdf5(f, 'SCNIC.module_analysis.write_modules_to_dir')


def write_modules_to_file(modules, path_str='modules.txt'):
    """
    Write module definitions to a tab-delimited file.

    Parameters
    ----------
    modules : dict
    path_str : str, default 'modules.txt'
    """
    # write all modules to file
    with open(path_str, 'w') as f:
        for module_, otus in modules.items():
            f.write('%s\t%s\n' % (module_, '\t'.join([str(otu) for otu in otus])))


def add_modules_to_metadata(modules, metadata):
    """
    Annotate metadata dictionary with module membership.

    Parameters
    ----------
    modules : dict
    metadata : dict

    Returns
    -------
    dict
    """
    for module_, otus in modules.items():
        for otu in otus:
            if str(otu) in metadata:
                metadata[str(otu)]['module'] = module_
            else:
                metadata[str(otu)] = dict()
                metadata[str(otu)]['module'] = module_
    return metadata
