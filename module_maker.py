import argparse
import os

import networkx as nx
import numpy as np

from biom import load_table, Table
from scipy.stats import spearmanr, rankdata, pearsonr
from collections import defaultdict
import matplotlib.pyplot as plt

def bh_adjust(p_vals):
    p_vals = np.array(p_vals)
    return p_vals*len(p_vals)/rankdata(p_vals)
    
def bonferroni_adjust(p_vals):
    return [i*len(p_vals) for i in p_vals]

def print_delimited(out_fp, lines, header=None):
    out = open(out_fp, 'w')
    if header != None:
        map(str, header)
        out.write('\t'.join(header)+'\n')
    for line in lines:
        out.write('\t'.join(map(str, line))+'\n')
    out.close()

def plot_networkx(G):
    graph_pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos, width=2, alpha=0.3, edge_color='green')
    nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')

    plt.show()

def filter_rel_abund(table, min_samples, min_percent):
    # first sample filter
    if min_samples != None:
        table.filter([i for i in table.ids(axis='observation') if sum(table.data(i, axis='observation') != 0) > 10], axis='observation')

    if min_percent != None:
        # set values below min_percent to zero
        for i in xrange(table.shape[0]):
            for j in xrange(table.shape[1]):
                if table.matrix_data[i,j] < args.min_percent:
                    table.matrix_data[i,j] = 0

        # filter out rows of all zeroes
        table.filter([i for i in table.ids(axis='observation') if sum(table.data(i, axis='observation'))!=0], axis='observation')

        # second sample filter
        if min_samples != None:
            table.filter([i for i in table.ids(axis='observation') if sum(table.data(i, axis='observation') != 0) > 10], axis='observation')

    return table

def paired_correlations_from_table(table, correl_method=spearmanr, p_adjust=bh_adjust):
    """Takes a biom table and finds correlations between all pairs of otus."""
    otus = table.ids(axis='observation')
    
    correls = list()

    # do all correlations
    for i in xrange(1,len(otus)):
    	otu_i = table.data(otus[i], axis='observation')
    	for j in xrange(i+1,len(otus)):
        	otu_j = table.data(otus[j], axis='observation')
        	correl = correl_method(otu_i, otu_j)
            	correls.append([otus[i], otus[j], correl[0], correl[1]])

    # adjust p-value if desired
    if p_adjust != None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust != None:
        header.append('adjusted_p')

    return correls, header

def make_net_from_correls(correls, min_p=.05):
    # filter to only include significant correlations
    correls = list(i for i in correls if i[3] < min_p and i[2] > 0)
    
    G = nx.Graph()
    for correl in correls:
        G.add_node(correl[0])
        G.add_node(correl[1])
        G.add_edge(correl[0],correl[1], r=correl[2],p=correl[3],p_adj=correl[4])
    return G

def make_modules(G, k=[2,3,4,5,6]):
    communities = dict()
    for k_val in list(k):
        cliques = list(nx.k_clique_communities(G, k_val))
        cliques = map(list, cliques)
        communities[k_val] = cliques
        for i,clique in enumerate(cliques):
            for node in clique:
                G.node[node]['k_'+str(k_val)] = i
    return G, {k: list(v) for k, v in communities.iteritems()}

def collapse_modules(table, cliques):
    in_module = set()
    modules = np.zeros((len(cliques),table.shape[1]))
    
    # for each clique merge values
    for i, clique in enumerate(cliques):
        in_module = in_module | set(clique)
        for feature in clique:
            modules[i] += table.data(feature, axis='observation')
    table.filter(in_module, axis = 'observation')
    
    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), modules))
    new_table_obs = list(table.ids(axis='observation'))+["module_"+str(i) for i in range(0,len(cliques))]
    return Table(new_table_matrix, new_table_obs, table.ids())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="location of input biom file")
    parser.add_argument("-o", "--output", help="output file location")
    parser.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    parser.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    parser.add_argument("-s", "--min_sample", help="minimum number of samples present in")
    parser.add_argument("-n", "--min_percent", help="minimum precent relative abundance of sample", type=float)
    args = parser.parse_args()
    
    # correlation and p-value adjustment methods
    correl_methods = {'spearman':spearmanr, 'pearson':pearsonr}
    p_methods = {'bh':bh_adjust, 'bon':bonferroni_adjust}
    correl_method = correl_methods[args.correl_method]
    if args.p_adjust != None:
        p_adjust = p_methods[args.p_adjust]
    
    # get features to be correlated
    table = load_table(args.input)
    
    # make new output directory and change to it
    os.makedirs(args.output)
    os.chdir(args.output)
    
    # convert to relative abundance and filter
    rel_table = table.norm(inplace=False)
    rel_table = filter_rel_abund(rel_table, args.min_sample, args.min_percent)
    
    # correlate feature
    correls, correl_header = paired_correlations_from_table(rel_table, correl_method, p_adjust)
    print_delimited('correls.txt', correls, correl_header)
    
    # make correlation network
    net = make_net_from_correls(correls)
    net.graph['correl_method'] = str(args.correl_method)
    net.graph['p_adjust'] = str(args.p_adjust)
    
    # make modules
    net, cliques = make_modules(net)
    
    # print network
    nx.write_gml(net, 'conetwork.gml')
    
    # collapse modules
    coll_table = collapse_modules(table, cliques[3])
    
    # print new table
    coll_table.to_json('make_modules.py', open('collapsed.biom', 'w'))

if __name__ == '__main__':
    main()
