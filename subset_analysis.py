from biom import load_table
from itertools import combinations
import random
from module_maker import *
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle
import argparse


def main1():
    table = load_table("HIV/otu_table_neg.biom")
    table.norm()
    table = filter_rel_abund(table, min_samples=8)
    samps = table.ids()
    edges = list()

    for i in combinations(range(0, 25), 20):
        sub_table = table.filter(samps[list(i)], inplace=False)
        correls, correl_header = \
            paired_correlations_from_table(sub_table, spearmanr, bh_adjust)
        net = make_net_from_correls(correls)
        edges.append(net.edges())

def make_master_edges(table):
    """input count table"""
    table.norm()
    master_table = filter_rel_abund(table, min_samples=table.shape[1]/3) # require samples to exist in 1/3 of samples
    master_correls, head = paired_correlations_from_table(master_table, spearmanr, bh_adjust)
    master_net = make_net_from_correls(master_correls)
    return master_net.edges()

def subsample_trees(table, reps, k, pkl_out = 'subsampled_edges.pkl'):
    """input count table, number of subsamples, size of subsamples, pkl output"""
    table.norm()
    
    # subsample
    samps = table.ids()
    edge_counts = Counter()
    combs = list(combinations(range(0, table.shape[1]), k))
    combs = random.sample(combs, reps)

    for count, i in enumerate(combs):
        sub_table = table.filter(samps[list(i)], inplace=False)
        # require observations to be in atleast 1/3 of samples
        sub_table = filter_rel_abund(table, min_samples=table.shape[1]/3)
        correls, correl_header = \
            paired_correlations_from_table(sub_table, spearmanr, bh_adjust)
        net = make_net_from_correls(correls)
        edge_counts.update(net.edges())
        print count    

    # dump edges found to file
    with open(pkl_out, 'wb') as pkl:
        cPickle.dump(edge_counts, pkl)
    
    return edge_counts

def bootstrap_trees(table, reps, pkl_out = 'bootstrapped_edges.pkl'):
    pass

def make_edge_plots(master_edges, edge_counts):
    # make plot of all edges
    plt.hist(edge_counts.values())
    plt.title("Distriubtion of all Found Edges")
    plt.xlabel("Number of Times Edge Found")
    plt.ylabel("Frequency")
    plt.savefig('all_edges.png', bbox_inches='tight')
    plt.clf()
    
    # make plot of master edges
    edge_counts_in_master = {k:edge_counts[k] for k in edge_counts if k in master_edges}
    if len(edge_counts_in_master) > 0:
        plt.hist(edge_counts_in_master.values())
        plt.title("Distribution of Edges in Master")
        plt.xlabel("Number of Times Edge Found")
        plt.ylabel("Frequency")
        plt.savefig('master_edges.png', bbox_inches='tight')
        plt.clf()
    
    # make plot of non-master edges
    edge_counts_not_master = {k:edge_counts[k] for k in edge_counts if k not in master_edges}
    if len(edge_counts_not_master) > 0:
        plt.hist(edge_counts_not_master.values())
        plt.title("Distribution of Edges in not Master")
        plt.xlabel("Number of Times Edge Found")
        plt.ylabel("Frequency")
        plt.savefig('nonmaster_edges.png', bbox_inches='tight')
        plt.clf()
    
    # make plot of overlapping
    plt.hist(edge_counts.values())
    if len(edge_counts_in_master) > 0:
        plt.hist(edge_counts_in_master.values())
    if len(edge_counts_not_master) > 0:
        plt.hist(edge_counts_not_master.values())
    plt.title("Distriubtion of Edges")
    plt.xlabel("Number of Times Edge Found")
    plt.ylabel("Frequency")
    plt.savefig('edges_overlap.png', bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--table", help="location of input biom file")
    parser.add_argument("-o", "--output", help="new output folder")
    parser.add_argument("-r", "--reps", help="repetitions", default=100, type=int)
    parser.add_argument("-a", "--sub_size", help="sub-sample size", type=int)
    
    args = parser.parse_args()
    
    table = load_table(args.table) # "HIV/otu_table_neg.biom"
    
    # make new output directory and change to it
    os.makedirs(args.output)
    os.chdir(args.output)
    
    edge_counts = subsample_trees(table, args.reps, args.sub_size)
    master_edges = make_master_edges(table)
    make_edge_plots(master_edges, edge_counts)

if __name__ == '__main__':
    main()
