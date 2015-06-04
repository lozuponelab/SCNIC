__author__ = 'shafferm'

"""functions used widely"""

from scipy.stats import rankdata
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
