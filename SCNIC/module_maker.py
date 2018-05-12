"""Make modules of observations based on cooccurence networks and collapse table"""

from __future__ import division

from SCNIC import general

import numpy as np
import pandas as pd
from biom import load_table, Table
import os
from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import squareform
from skbio.tree import TreeNode
from itertools import combinations
import networkx as nx


def correls_to_cor(correls, metric='r'):
    # convert to square
    cor = correls.unstack()[metric]
    cor = cor.reindex(cor.columns)
    # fill na's
    for otu_i, otu_j in combinations(cor.index, 2):
        if pd.isna(cor.loc[otu_i, otu_j]):
            cor.loc[otu_i, otu_j] = cor.loc[otu_j, otu_i]
        else:
            cor.loc[otu_j, otu_i] = cor.loc[otu_i, otu_j]
    # for otu in cor.index:
    #     cor.loc[otu, otu] = 1
    return squareform(cor, checks=False), cor.index


def cor_to_dist(cor):
    # convert from correlation to distance
    return 1 - ((cor + 1) / 2)


def make_modules(dist, min_dist, obs_ids):
    # create linkage matrix using complete linkage
    Z = complete(dist)
    # make tree from linkage matrix with names from dist
    tree = TreeNode.from_linkage_matrix(Z, obs_ids)
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
            modules = sorted(modules, key=len, reverse=True)
            return modules
    raise ValueError("Well, how did I get here?")


def collapse_modules(table, modules, prefix="module"):
    """collapse created modules in a biom table, members of multiple modules will be added to the smallest module"""
    table = table.copy()
    module_array = np.zeros((len(modules), table.shape[1]))

    seen = set()
    for i, module in enumerate(modules):
        seen = seen | module
        # sum everything in the module
        module_array[i] = np.sum([table.data(feature, axis="observation") for feature in module], axis=0)

    table.filter(seen, axis='observation', invert=True)

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), module_array))
    new_table_obs = list(table.ids(axis='observation')) + ['_'.join([prefix, str(i)]) for i in range(len(modules))]
    return Table(new_table_matrix, new_table_obs, table.ids())


def write_modules_to_dir(table, modules):
    # for each module merge values and print modules to file
    if not os.path.isdir("modules"):
        os.makedirs("modules")
    # reverse modules so observations will be added to smallest modules
    for i, module in enumerate(modules):
        # make biom tables for each module and write to file
        module_table = table.filter(module, axis='observation', inplace=False)
        module_table.to_json("modulemaker.py", open("modules/%s.biom" % i, 'w'))


def write_modules_to_file(modules, prefix="module"):
    # write all modules to file
    with open("modules.txt", 'w') as f:
        for i, module in enumerate(modules):
            f.write('_'.join([prefix, str(i)]) + '\t' + '\t'.join([str(j) for j in module]) + '\n')


def module_maker(args):
    logger = general.Logger("SCNIC_module_log.txt")
    logger["SCNIC analysis type"] = "module"

    # read in correlations file
    correls = pd.read_table(args.input, index_col=(0, 1), sep='\t', dtype={'feature1': str, 'feature2': str})
    logger["input correls"] = args.input
    if args.verbose:
        print("correls.txt read")

    # sanity check args
    if args.min_r is not None and args.min_p is not None:
        raise ValueError("arguments min_p and min_r may not be used concurrently")
    if args.min_r is None and args.min_p is None:
        raise ValueError("argument min_p or min_r must be used")

    # read in correlations file and make distance matrix
    if args.min_r is not None:
        min_dist = cor_to_dist(args.min_r)
        logger["minimum r value"] = args.min_r
        cor, labels = correls_to_cor(correls)
        dist = cor_to_dist(cor)
    if args.min_p is not None:
        # TODO: This
        raise NotImplementedError()

    # read in biom table if given
    if args.table is not None:
        table = load_table(args.table)
        logger["input uncollapsed table"] = args.table
        if args.verbose:
            print("otu table read")

    # make new output directory and change to it
    if args.output is not None:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        os.chdir(args.output)
    logger["output directory"] = os.getcwd()

    # make modules
    modules = make_modules(dist, min_dist, obs_ids=labels)
    logger["number of modules created"] = len(modules)
    if args.verbose:
        print("Modules Formed")
        print("number of modules: %s" % len(modules))
        print("number of observations in modules: %s" % np.sum([len(i) for i in modules]))
        print("")
    write_modules_to_file(modules)

    # collapse modules
    if args.table is not None:
        coll_table = collapse_modules(table, modules)
        write_modules_to_dir(table, modules)
        logger["number of observations in output table"] = coll_table.shape[0]
        if args.verbose:
            print("Table Collapsed")
            print("collapsed Table Observations: " + str(coll_table.shape[0]))
            print("")
        coll_table.to_json('make_modules.py', open('collapsed.biom', 'w'))

    # make network
    if args.table is not None:
        metadata = general.get_metadata_from_table(table)
        net = general.correls_to_net(correls, conet=True, metadata=metadata, min_p=args.min_p, min_r=args.min_r)
    else:
        net = general.correls_to_net(correls, conet=True, min_p=args.min_p, min_r=args.min_r)

    nx.write_gml(net, 'correlation_network.gml')
    if args.verbose:
        print("Network Generated")
        print("number of nodes: %s" % str(net.number_of_nodes()))
        print("number of edges: %s" % str(net.number_of_edges()))
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()

    logger.output_log()
