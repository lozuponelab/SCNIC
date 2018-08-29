"""Make modules of observations based on cooccurence networks and collapse table"""
from collections import defaultdict

import numpy as np
import pandas as pd
from biom import load_table
from biom.util import biom_open
import os
import networkx as nx


from SCNIC import general
from SCNIC import module_analysis as ma


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
        min_dist = ma.cor_to_dist(args.min_r)
        logger["minimum r value"] = args.min_r
        cor, labels = ma.correls_to_cor(correls)
        dist = ma.cor_to_dist(cor)
    elif args.min_p is not None:
        # TODO: This
        raise NotImplementedError()
    else:
        raise ValueError("this is prevented above")

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
    modules = ma.make_modules(dist, min_dist, obs_ids=labels)
    logger["number of modules created"] = len(modules)
    if args.verbose:
        print("Modules Formed")
        print("number of modules: %s" % len(modules))
        print("number of observations in modules: %s" % np.sum([len(i) for i in modules]))
        print("")
    ma.write_modules_to_file(modules)

    # collapse modules
    if args.table is not None:
        coll_table = ma.collapse_modules(table, modules)
        ma.write_modules_to_dir(table, modules)
        logger["number of observations in output table"] = coll_table.shape[0]
        if args.verbose:
            print("Table Collapsed")
            print("collapsed Table Observations: " + str(coll_table.shape[0]))
            print("")
        with biom_open('collapsed.biom', 'w') as f:
            coll_table.to_hdf5(f, 'make_modules.py')

    # make network
    if args.table is not None:
        metadata = general.get_metadata_from_table(table)
    else:
        metadata = defaultdict(dict)
    metadata = ma.add_modules_to_metadata(modules, metadata)
    correls_filter = general.filter_correls(correls, conet=True, min_p=args.min_p, min_r=args.min_r)
    net = general.correls_to_net(correls_filter, metadata=metadata)

    nx.write_gml(net, 'correlation_network.gml')
    if args.verbose:
        print("Network Generated")
        print("number of nodes: %s" % str(net.number_of_nodes()))
        print("number of edges: %s" % str(net.number_of_edges()))
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()

    logger.output_log()
