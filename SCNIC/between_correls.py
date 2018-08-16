"""
Workflow script for finding correlations between pairs of biom tables, making networks, finding modules and collapsing
modules.
"""
import os
from biom import load_table
from scipy.stats import spearmanr, pearsonr
import networkx as nx
import numpy as np
import shutil

from SCNIC import general
from SCNIC import correlation_analysis as ca

__author__ = 'shafferm'

# TODO: output heat map with clusters


def between_correls(args):
    """TABLES MUST SORT SO THAT SAMPLES ARE IN THE SAME ORDER """
    logger = general.Logger("SCNIC_log.txt")
    logger["SCNIC analysis type"] = "between"

    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr}
    correl_method = correl_methods[args.correl_method]

    # load tables
    table1 = load_table(args.table1)
    table2 = load_table(args.table2)
    logger["input table 1"] = args.table1
    logger["input table 1"] = args.table2

    table1 = table1.sort()
    table2 = table2.sort()

    # make new output directory and change to it
    if args.force and args.output is not None:
        shutil.rmtree(args.output, ignore_errors=True)
    if args.output is not None:
        os.makedirs(args.output)
        os.chdir(args.output)
        logger["output directory"] = args.output

    # filter tables
    if args.sparcc_filter is True:
        table1 = general.sparcc_paper_filter(table1)
        table2 = general.sparcc_paper_filter(table2)
        print("Table 1 filtered: %s observations" % str(table1.shape[0]))
        print("Table 2 filtered: %s observations" % str(table2.shape[0]))
        logger["sparcc paper filter"] = True
        logger["number of observations present in table 1 after filter"] = table1.shape[0]
        logger["number of observations present in table 2 after filter"] = table2.shape[0]
    if args.min_sample is not None:
        table1 = general.filter_table(table1, args.min_sample)
        table2 = general.filter_table(table2, args.min_sample)

    if not np.array_equal(table1.ids(), table2.ids()):
        raise ValueError("Tables have different sets of samples present")

    metadata = general.get_metadata_from_table(table1)
    metadata.update(general.get_metadata_from_table(table2))

    # make correlations
    logger["correlation metric"] = args.correl_method
    logger["p adjustment method"] = args.p_adjust
    correls = ca.between_correls_from_tables(table1, table2, correl_method, nprocs=args.procs)
    correls.sort_values(correls.columns[-1], inplace=True)
    correls['p_adj'] = general.p_adjust(correls['p'])
    correls.to_csv(open('correls.txt', 'w'), sep='\t', index=True)

    # make network
    correls_filt = general.filter_correls(correls, min_p=args.min_p, min_r=args.min_r)
    net = general.correls_to_net(correls_filt, metadata=metadata)
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()
    nx.write_gml(net, 'crossnet.gml')

    logger.output_log()
