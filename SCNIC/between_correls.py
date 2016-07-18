__author__ = 'shafferm'

from biom import load_table
from scipy.stats import spearmanr, pearsonr
import general
import os
import networkx as nx
import numpy as np


def between_correls_from_tables(table1, table2, correl_method=spearmanr, p_adjust=general.bh_adjust):
    correls = list()

    for data_i, otu_i, metadata_i in table1.iter(axis="observation"):
        for data_j, otu_j, metadata_j in table2.iter(axis="observation"):
            corr = correl_method(data_i, data_j)
            correls.append([otu_i, otu_j, corr[0], corr[1]])

    p_adjusted = p_adjust([i[3] for i in correls])
    for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    return correls, ['feature1', 'feature2', 'r', 'p', 'p_adj']


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
    table1 = general.filter_table(table1)
    metadata = general.get_metadata_from_table(table1)
    table2 = general.filter_table(table2)
    metadata.update(general.get_metadata_from_table(table2))

    # make correlations
    logger["correlation metric"] = args.correl_method
    logger["p adjustment method"] = args.p_adjust
    correls, correl_header = between_correls_from_tables(table1, table2, correl_method, p_adjust)
    general.print_delimited('correls.txt', correls, correl_header)

    # make network
    net = general.correls_to_net(correls, metadata=metadata, min_p=args.min_p)
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()
    nx.write_gml(net, 'crossnet.gml')

    logger.output_log()
    print('\a')
