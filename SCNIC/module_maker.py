"""Make modules of observations based on cooccurence networks and collapse table"""

from __future__ import division

import networkx as nx
import numpy as np
from biom import Table
import os


def make_modules(graph, k=3, prefix="module"):
    """make modules with networkx k-clique communities and annotate network"""
    premodules = list(nx.k_clique_communities(graph, k))
    # reverse modules so observations will be added to smallest modules
    premodules = list(enumerate(premodules))
    premodules.reverse()

    modules = dict()
    seen = set()
    for i, module in premodules:
        # process module
        module = module-seen
        seen = seen | module
        modules[prefix+"_"+str(i)] = module
        for node in module:
            graph.node[node][prefix] = i
    return graph, modules


def make_modules_multik(graph, k=None):
    """make modules with networkx k-clique communities and annotate network"""
    if k is None:
        k = [2, 3, 4, 5, 6]
    communities = dict()
    for k_val in list(k):
        cliques = list(nx.k_clique_communities(graph, k_val))
        cliques = [list(i) for i in cliques]
        communities[k_val] = cliques
        for i, clique in enumerate(cliques):
            for node in clique:
                graph.node[node]['k_' + str(k_val)] = i
    return graph, {k: list(v) for k, v in communities.iteritems()}


def collapse_modules(table, modules):
    """collapse created modules in a biom table, members of multiple modules will be added to the smallest module"""
    table = table.copy()
    module_array = np.zeros((len(modules), table.shape[1]))

    seen = set()
    for i, module in modules.iteritems():
        seen = seen | module
        # sum everything in the module
        module_array[int(i.split("_")[-1])] = np.sum([table.data(feature, axis="observation") for feature in module],
                                                     axis=0)

    table.filter(seen, axis='observation', invert=True)

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), module_array))
    new_table_obs = list(table.ids(axis='observation')) + modules.keys()
    return Table(new_table_matrix, new_table_obs, table.ids())


def write_modules_to_dir(table, modules):
    # for each module merge values and print modules to file
    os.makedirs("modules")
    # reverse modules so observations will be added to smallest modules
    for i, module in modules.iteritems():
        # make biom tables for each module and write to file
        module_table = table.filter(module, axis='observation', inplace=False)
        module_table.to_json("modulemaker.py", open("modules/%s.biom" % i, 'w'))


def write_modules_to_file(modules):
    # write all modules to file
    with open("modules.txt", 'w') as f:
        for i, module in modules.iteritems():
            f.write(i + '\t' + '\t'.join([str(j) for j in module]) + '\n')


def collapse_modules_multik(table, cliques, prefix="module_"):
    """collapse created modules in a biom table"""
    in_module = set()
    modules = np.zeros((len(cliques), table.shape[1]))

    # for each clique merge values
    for i, clique in enumerate(cliques):
        in_module = in_module | set(clique)
        for feature in clique:
            modules[i] += table.data(feature, axis='observation')
    table.filter(in_module, axis='observation')

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), modules))
    new_table_obs = list(table.ids(axis='observation')) + [prefix + str(i) for i in range(0, len(cliques))]
    return Table(new_table_matrix, new_table_obs, table.ids(), table.metadata(axis="observation"),
                 table.metadata(axis="sample"))
