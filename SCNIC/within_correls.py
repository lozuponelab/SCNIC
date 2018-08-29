from __future__ import division

import os

import networkx as nx

from biom import load_table
from scipy.stats import spearmanr, pearsonr, kendalltau

from SCNIC import general
from SCNIC import correlation_analysis as ca


def within_correls(args):
    logger = general.Logger("SCNIC_within_log.txt")
    logger["SCNIC analysis type"] = "within"

    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'kendall': kendalltau, 'sparcc': 'sparcc'}
    correl_method = correl_methods[args.correl_method.lower()]

    # get features to be correlated
    table = load_table(args.input)
    logger["input table"] = args.input
    if args.verbose:
        print("Table loaded: " + str(table.shape[0]) + " observations")
        print("")
    logger["number of samples in input table"] = table.shape[1]
    logger["number of observations in input table"] = table.shape[0]

    # make new output directory and change to it
    if args.output is not None:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        os.chdir(args.output)
    logger["output directory"] = os.getcwd()

    # filter
    if args.sparcc_filter is True:
        table_filt = general.sparcc_paper_filter(table)
        if args.verbose:
            print("Table filtered: %s observations" % str(table_filt.shape[0]))
            print("")
        logger["sparcc paper filter"] = True
        logger["number of observations present after filter"] = table_filt.shape[0]
    elif args.min_sample is not None:
        table_filt = general.filter_table(table, args.min_sample)
        if args.verbose:
            print("Table filtered: %s observations" % str(table_filt.shape[0]))
            print("")
        logger["min samples present"] = args.min_sample
        logger["number of observations present after filter"] = table_filt.shape[0]
    else:
        table_filt = table

    logger["number of processors used"] = args.procs

    # correlate features
    if correl_method in [spearmanr, pearsonr, kendalltau]:
        # calculate correlations
        if args.verbose:
            print("Correlating with %s" % args.correl_method)
        # correlate feature
        correls = ca.calculate_correlations(table_filt, correl_method, nprocs=args.procs)
    elif correl_method == 'sparcc':
        correls = ca.fastspar_correlation(table_filt, verbose=args.verbose, nprocs=args.procs)
        if args.sparcc_p is not None:
            raise NotImplementedError()  # TODO: reimplement with fastspar
    else:
        raise ValueError("How did this even happen?")
    logger["distance metric used"] = args.correl_method
    if args.verbose:
        print("Features Correlated")
        print("")

    if 'p' in correls.columns:
        correls['p_adj'] = general.p_adjust(correls['p'])
    correls.to_csv('correls.txt', sep='\t', index_label=('feature1', 'feature2'))
    if args.verbose:
        print("Correls.txt written")

    # make correlation network
    metadata = general.get_metadata_from_table(table_filt)
    net = general.correls_to_net(correls, metadata=metadata)
    nx.write_gml(net, 'correlation_network.gml')
    if args.verbose:
        print("Network made")
        print("")

    logger.output_log()
