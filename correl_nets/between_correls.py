__author__ = 'shafferm'

import argparse
from biom import load_table
from scipy.stats import spearmanr, pearsonr
from correl_nets import general
import os
import networkx as nx


def between_correls(table1, table2, correl_method=spearmanr, p_adjust=general.bh_adjust):
    otus1 = table1.ids(axis="observation")
    otus2 = table2.ids(axis="observation")

    correls = list()

    for otu1 in otus1:
        row1 = table1.data(otu1, axis="observation")
        for otu2 in otus2:
            row2 = table2.data(otu2, axis="observation")
            corr = correl_method(row1, row2)
            correls.append([otu1, otu2, corr[1], corr[2]])

    p_adjusted = p_adjust([i[3] for i in correls])
    correls = [correls[i].append(p_adjusted[i]) for i in correls]
    #for i in xrange(len(correls)):
    #    correls[i].append(p_adjusted[i])

    return correls, ['feature1', 'feature2', 'r', 'p', 'p_adj']


def make_net_from_correls(correls, min_p=.05):
    """make network from set of correlations with values less than a minimum"""
    # filter to only include significant correlations
    try:
        correls = list(i for i in correls if i[4] < min_p)
    except IndexError:
        correls = list(i for i in correls if i[3] < min_p)

    graph = nx.Graph()
    for correl in correls:
        graph.add_node(correl[0])
        graph.add_node(correl[1])
        graph.add_edge(correl[0], correl[1], r=correl[2],
                       p=correl[3], p_adj=correl[4], sign_pos=abs(correl[2]) == correl[2])
    return graph


def main(args):

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

    # make new output directory and change to it
    os.makedirs(args.output)
    os.chdir(args.output)

    # filter tables
    table1 = general.filter_table(table1)
    table2 = general.filter_table(table2)

    # make correlations
    correls, correl_header = between_correls(table1, table2, correl_method, p_adjust)
    general.print_delimited('correls.txt', correls, correl_header)

    # make network
    net = make_net_from_correls(correls)
    nx.write_gml(net, 'crossnet.gml')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-1", "--table1", help="table to be correlated")
    parser.add_argument("-2", "--table2", help="second table to be correlated")
    parser.add_argument("-o", "--output", help="output file location")
    parser.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    parser.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    parser.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)

    main(parser.parse_args())
