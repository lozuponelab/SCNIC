"""
Workflow script for finding correlations between pairs of biom tables, making networks, finding modules and collapsing
modules.
"""
import os
from biom import load_table
from scipy.stats import spearmanr, pearsonr
import networkx as nx
import numpy as np
from SCNIC import general
from SCNIC.correlation_analysis import between_correls_from_tables

__author__ = 'shafferm'


def between_correls(args):
    """TABLES MUST SORT SO THAT SAMPLES ARE IN THE SAME ORDER """
    logger = general.Logger("SCNIC_log.txt")
    logger["SCNIC analysis type"] = "between"

    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr}
    p_methods = {'bh': general.bh_adjust, 'bon': general.bonferroni_adjust}
    correl_method = correl_methods[args.correl_method]
    if args.p_adjust is not None:
        p_adjust = p_methods[args.p_adjust]
    else:
        p_adjust = None

    # load tables
    table1 = load_table(args.table1)
    table2 = load_table(args.table2)
    logger["input table 1"] = args.table1
    logger["input table 1"] = args.table2

    table1 = table1.sort()
    table2 = table2.sort()

    if not np.array_equal(table1.ids(), table2.ids()):
        raise ValueError("Tables have different sets of samples present")

    # make new output directory and change to it
    if args.output is not None:
        os.makedirs(args.output)
        os.chdir(args.output)
        logger["output directory"] = args.output

    # filter tables
    if args.min_sample is not None:
        table1 = general.filter_table(table1, args.min_sample)
        metadata = general.get_metadata_from_table(table1)
        table2 = general.filter_table(table2, args.min_sample)
        metadata.update(general.get_metadata_from_table(table2))
    else:
        metadata = general.get_metadata_from_table(table1)
        metadata.update(general.get_metadata_from_table(table2))

    # make correlations
    logger["correlation metric"] = args.correl_method
    logger["p adjustment method"] = args.p_adjust
    correls = between_correls_from_tables(table1, table2, correl_method)
    correls.sort_values(correls.columns[-1], inplace=True)
    correls.to_csv(open('correls.txt', 'w'), sep='\t', index=False)

    # adjust p-values
    correls['p_adj'] = p_adjust(correls['p'])

    # make network
    net = general.correls_to_net(correls, metadata=metadata, min_p=args.min_p, min_r=args.min_r)
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()
    nx.write_gml(net, 'crossnet.gml')

    logger.output_log()
    print '\a'
