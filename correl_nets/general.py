__author__ = 'shafferm'

"""functions used widely"""
# TODO: Make correl class and implement across package

from collections import defaultdict

from scipy.stats import rankdata
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_metadata_from_table(table):
    metadata = defaultdict(dict)
    for obs in table.ids(axis="observtion"):
        metadata[obs] = table.metadata(obs, axis="observation")
    return metadata


def bh_adjust(p_vals):
    """benjamini-hochberg p-value adjustment"""
    p_vals = np.array(p_vals)
    return p_vals*len(p_vals)/rankdata(p_vals, method='dense')


def bonferroni_adjust(p_vals):
    """bonferroni p-value adjustment"""
    return [i*len(p_vals) for i in p_vals]


def print_delimited(out_fp, lines, header=None):
    """print a tab delimited file with optional header"""
    out = open(out_fp, 'w')
    if header is not None:
        out.write('\t'.join([str(i) for i in header])+'\n')
    for line in lines:
        out.write('\t'.join([str(i) for i in line])+'\n')
    out.close()


def read_delimited(in_fp, header=False):
    """Read in delimited file"""
    delim = list()
    with open(in_fp) as f:
        if header is True:
            f.readline()
        for line in f:
            delim.append(line.strip().split('\t'))
    return delim


def correls_to_net(correls, min_p=.05, conet=False, metadata=None):
    # filter to only include significant correlations
    if conet:
        try:
            correls = list(i for i in correls if i[4] < min_p and i[2] > 0)
        except IndexError:
            correls = list(i for i in correls if i[3] < min_p and i[2] > 0)
    else:
        try:
            correls = list(i for i in correls if i[4] < min_p)
        except IndexError:
            correls = list(i for i in correls if i[3] < min_p)
    graph = nx.Graph()
    for correl in correls:
        graph.add_node(correl[0])
        try:
            for key in metadata:
                graph.node[correl[0]][key] = metadata[correl[0]][key]
        except:
            pass
        graph.add_node(correl[1])
        try:
            for key in metadata:
                graph.node[correl[1]][key] = metadata[correl[1]][key]
        except:
            pass
        graph.add_edge(correl[0], correl[1], r=correl[2],
                       p=correl[3], p_adj=correl[4], sign_pos=abs(correl[2]) == correl[2])
    return graph


def make_net_from_correls(correls_fp, conet=False):
    correls = read_delimited(correls_fp, header=True)
    for i, correl in enumerate(correls):
        for j in xrange(2,len(correl)):
            correls[i][j] = float(correls[i][j])
    if conet:
        return correls_to_conet(correls)
    else:
        return correls_to_net(correls)


def filter_table(table, min_samples=None, to_file=False):
    """filter relative abundance table, by default throw away things greater than 1/3 zeros"""
    table = table.copy()
    # first sample filter
    if min_samples is not None:
        to_keep = [i for i in table.ids(axis='observation')
                   if sum(table.data(i, axis='observation') != 0) >= min_samples]
    else:
        to_keep = [i for i in table.ids(axis='observation')
                   if sum(table.data(i, axis='observation') != 0) >= table.shape[1]/3]
    table.filter(to_keep, axis='observation')

    if to_file:
        table.to_json('filter_table', open("filtered_tab.biom", 'w'))
        # open("filtered_rel_abund.txt", 'w').write(table.to_tsv())

    return table


def plot_networkx(graph):
    """plot networkx object in matplotlib"""
    graph_pos = nx.circular_layout(graph)

    nx.draw_networkx_nodes(graph, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(graph, graph_pos, width=2, alpha=0.3, edge_color='green')
    nx.draw_networkx_labels(graph, graph_pos, font_size=12, font_family='sans-serif')

    plt.show()
