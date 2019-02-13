"""Make modules of observations based on cooccurence networks and collapse table"""
from collections import defaultdict

import numpy as np
import pandas as pd
from biom import load_table
from biom.util import biom_open
from os import path
import os
import networkx as nx


from SCNIC import general
from SCNIC import module_analysis as ma


def module_maker(input_loc, output_loc, min_p=None, min_r=None, method='naive', k_size=3, gamma=.4, table_loc=None,
                 prefix='module', verbose=False):
    logger = general.Logger(path.join(output_loc, "SCNIC_module_log.txt"))
    logger["SCNIC analysis type"] = "module"

    # read in correlations file
    correls = pd.read_csv(input_loc, index_col=(0, 1), sep='\t')
    correls.index = pd.MultiIndex.from_tuples([(str(id1), str(id2)) for id1, id2 in correls.index])
    logger["input correls"] = input_loc
    if verbose:
        print("correls.txt read")

    # sanity check args
    if min_r is not None and min_p is not None:
        raise ValueError("arguments min_p and min_r may not be used concurrently")
    if min_r is None and min_p is None:
        raise ValueError("argument min_p or min_r must be used")

    # make new output directory and change to it
    if output_loc is not None:
        if not path.isdir(output_loc):
            os.makedirs(output_loc)
    logger["output directory"] = path.abspath(output_loc)

    # make modules
    if method == 'naive':
        modules = ma.make_modules_naive(correls, min_r, min_p, prefix=prefix)
    elif method == 'k_cliques':
        modules = ma.make_modules_k_cliques(correls, min_r, min_p, k_size, prefix=prefix)
    elif method == 'louvain':
        modules = ma.make_modules_louvain(correls, min_r, min_p, gamma, prefix=prefix)
    else:
        raise ValueError('%s is not a valid module picking method' % method)
    logger["number of modules created"] = len(modules)
    if verbose:
        print("Modules Formed")
        print("number of modules: %s" % len(modules))
        print("number of observations in modules: %s" % np.sum([len(i) for i in modules]))
        print("")
    ma.write_modules_to_file(modules, path_str=path.join(output_loc, 'modules.txt'))

    # collapse modules
    if table_loc is not None:
        table = load_table(table_loc)
        logger["input uncollapsed table"] = table_loc
        if verbose:
            print("otu table read")
        coll_table = ma.collapse_modules(table, modules)
        # ma.write_modules_to_dir(table, modules)
        logger["number of observations in output table"] = coll_table.shape[0]
        if verbose:
            print("Table Collapsed")
            print("collapsed Table Observations: " + str(coll_table.shape[0]))
            print("")
        with biom_open(path.join(output_loc, 'collapsed.biom'), 'w') as f:
            coll_table.to_hdf5(f, 'make_modules.py')
        metadata = general.get_metadata_from_table(table)
    else:
        metadata = defaultdict(dict)

    # make network
    metadata = ma.add_modules_to_metadata(modules, metadata)
    correls_filter = general.filter_correls(correls, conet=True, min_p=min_p, min_r=min_r)
    net = general.correls_to_net(correls_filter, metadata=metadata)

    nx.write_gml(net, path.join(output_loc, 'correlation_network.gml'))
    if verbose:
        print("Network Generated")
        print("number of nodes: %s" % str(net.number_of_nodes()))
        print("number of edges: %s" % str(net.number_of_edges()))
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()

    logger.output_log()
