"""
Module creation and collapse workflow.

This module builds modules from correlation networks and optionally collapses
a BIOM table by module membership.
"""

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


def module_maker(input_loc, output_loc, max_p=None, min_r=None, method='naive', k_size=3, gamma=.4, table_loc=None,
                 prefix='module', verbose=False):
    """
    Create modules from correlation data and optionally collapse a BIOM table.

    Parameters
    ----------
    input_loc : str
    output_loc : str
    max_p : float or None
    min_r : float or None
    method : str, default 'naive'
    k_size : int, default 3
    gamma : float, default .4
    table_loc : str or None
    prefix : str, default 'module'
    verbose : bool, default False
    """
    logger = general.Logger(path.join(output_loc, "SCNIC_module_log.txt"))
    logger["SCNIC analysis type"] = "module"

    # read in correlations file
    correls = pd.read_csv(input_loc, index_col=(0, 1), sep='\t')
    correls.index = pd.MultiIndex.from_tuples([(str(id1), str(id2)) for id1, id2 in correls.index])
    logger["input correls"] = input_loc
    if verbose:
        print(f"{input_loc} read")

    # make new output directory and change to it
    if output_loc is not None:
        if not path.isdir(output_loc):
            os.makedirs(output_loc)
    logger["output directory"] = path.abspath(output_loc)

    ## save which module creation method was used to the log file 
    logger["module creation method"] = method

    # sanity check args
    if min_r is not None and max_p is not None:
        raise TypeError("arguments max_p and min_r may not be used concurrently")
    if min_r is None and max_p is None:
        raise TypeError("argument max_p or min_r must be used")

    ## save either min_r or max_p threshold values to the log file
    if min_r is not None:
        logger["minimum r-value threshold"] = min_r
    elif max_p is not None:
        logger["maximum p-value threshold"] = max_p

    # make modules
    if method == 'naive':
        modules = ma.make_modules_naive(correls, min_r, max_p, prefix=prefix)
    elif method == 'k_cliques':
        modules = ma.make_modules_k_cliques(correls, min_r, max_p, k_size, prefix=prefix)
        logger["k size"] = k_size
    elif method == 'louvain':
        modules = ma.make_modules_louvain(correls, min_r, max_p, gamma, prefix=prefix)
        logger["gamma value"] = gamma
    else:
        raise KeyError('%s is not a valid module creation method' % method)
    logger["number of modules created"] = len(modules)
    if verbose:
        print(f"Modules formed using {method}!")
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
            print("Table collapsed!")
            print("collapsed table observations: " + str(coll_table.shape[0]))
            print("")
        with biom_open(path.join(output_loc, 'collapsed.biom'), 'w') as f:
            coll_table.to_hdf5(f, 'make_modules.py')
        metadata = general.get_metadata_from_table(table)
    else:
        metadata = defaultdict(dict)

    # make network
    metadata = ma.add_modules_to_metadata(modules, metadata)
    correls_filter = general.filter_correls(correls, max_p=max_p, min_r=min_r, conet=True)
    net = general.correls_to_net(correls_filter, metadata=metadata)

    nx.write_gml(net, path.join(output_loc, 'module_correlation_network.gml'))
    if verbose:
        print("Network made!")
        print("number of nodes: %s" % str(net.number_of_nodes()))
        print("number of edges: %s" % str(net.number_of_edges()))
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()

    logger.output_log()
