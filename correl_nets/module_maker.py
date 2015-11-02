"""Make modules of observations based on cooccurence networks and collapse table"""


##TODO: Add parameters log output file to output folder

import os
import shutil
import glob
import sys

import networkx as nx
import numpy as np
import pysurvey as ps

import general
from biom import load_table, Table
from biom.exception import UnknownIDError
from scipy.stats import spearmanr, pearsonr
from pysurvey import SparCC as sparcc
from functools import partial

#used by experimental functions
from scipy.spatial.distance import squareform
import time
def paired_correlations_from_table(table, correl_method=spearmanr, p_adjust=general.bh_adjust):
    """Takes a biom table and finds correlations between all pairs of otus."""
    correls = list()

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in table.iter_pairwise(axis='observation'):
        correl = correl_method(data_i, data_j)
        correls.append([str(otu_i), str(otu_j), correl[0], correl[1]])

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust is not None:
        header.append('adjusted_p')

    return correls, header


def boostrapped_correlation_multi(tup, temp_folder, cor_temp):
    """tup is a tuple where tup[0] is num and tup[1] is the file name"""
    df = ps.read_txt(tup[1], verbose=False)
    cor = ps.basis_corr(df, oprint=False)[0]
    ps.write_txt(cor, temp_folder+cor_temp+str(tup[0])+".txt", T=False)


def sparcc_correlations_multi(table, p_adjust=general.bh_adjust, temp_folder=os.getcwd()+"/temp/",
                              boot_temp="bootstrap_", cor_temp="cor_", table_temp="temp_table.txt",
                              bootstraps=100):
    """Calculate correlations with sparcc"""
    import multiprocessing

    os.mkdir(temp_folder)
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)

    # make tab delimited, delete first line and reread in
    with open(temp_folder+table_temp, 'w') as f:
        f.write('\n'.join(table.to_tsv().split("\n")[1:]))
    df = ps.read_txt(temp_folder+table_temp, verbose=False)

    sparcc.make_bootstraps(df, bootstraps, boot_temp+"#.txt", temp_folder)
    cor, cov = ps.basis_corr(df, oprint=False)

    pfun = partial(boostrapped_correlation_multi, temp_folder=temp_folder, cor_temp=cor_temp)
    tups = enumerate(glob.glob(temp_folder+boot_temp+"*.txt"))
    pool.map(pfun, tups)
    pool.close()
    pool.join()

    p_vals = sparcc.get_pvalues(cor, temp_folder+cor_temp+"#.txt", bootstraps)
    # generate correls
    correls = list()
    for i in xrange(len(cor.index)):
        for j in xrange(i+1, len(cor.index)):
            correls.append([str(cor.index[i]), str(cor.index[j]), cor.iat[i, j], p_vals.iat[i, j]])

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust is not None:
        header.append('adjusted_p')

    # cleanup, remove all of bootstraps folder
    shutil.rmtree(temp_folder)

    return correls, header


def boostrapped_correlation(in_file, temp_folder, cor_temp, num):
    df = ps.read_txt(in_file, verbose=False)
    cor = ps.basis_corr(df, oprint=False)[0]
    ps.write_txt(cor, temp_folder+cor_temp+str(num)+".txt", T=False)


def sparcc_correlations(table, p_adjust=general.bh_adjust, temp_folder=os.getcwd()+"/temp/",
                              boot_temp="bootstrap_", cor_temp="cor_", table_temp="temp_table.txt",
                              bootstraps=100):
    """"""
    # setup
    os.mkdir(temp_folder)

    # make tab delimited, delete first line and reread in
    with open(temp_folder+table_temp, 'w') as f:
        f.write('\n'.join(table.to_tsv().split("\n")[1:]))
    df = ps.read_txt(temp_folder+table_temp, verbose=False)

    # calculate correlations
    cor, cov = ps.basis_corr(df, oprint=False)

    # calculate p-values
    sparcc.make_bootstraps(df, bootstraps, boot_temp+"#.txt", temp_folder)
    for i, file in enumerate(glob.glob(temp_folder+boot_temp+"*.txt")):
        boostrapped_correlation(file, temp_folder, cor_temp, i)

    p_vals = sparcc.get_pvalues(cor, temp_folder+cor_temp+"#.txt", bootstraps)
    # generate correls
    correls = list()
    for i in xrange(len(cor.index)):
        for j in xrange(i+1, len(cor.index)):
            correls.append([str(cor.index[i]), str(cor.index[j]), cor.iat[i, j], p_vals.iat[i, j]])

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust is not None:
        header.append('adjusted_p')

    # cleanup, remove all of bootstraps folder
    shutil.rmtree(temp_folder)
    return correls, header


def boostrapped_correlation_lowmem(in_file):
    df = ps.read_txt(in_file, verbose=False)
    cor = ps.basis_corr(df, oprint=False)[0]
    cor = squareform(cor, checks=False)
    return cor


def sparcc_correlations_lowmem(table, p_adjust=general.bh_adjust, temp_folder=os.getcwd()+"/temp/",
                              boot_temp="bootstrap_", table_temp="temp_table.txt",
                              bootstraps=100):
    """"""
    # setup
    os.mkdir(temp_folder)

    # make tab delimited, delete first line and reread in
    # TODO: Convert to making pandas dataframe directly
    with open(temp_folder+table_temp, 'w') as f:
        f.write('\n'.join(table.to_tsv().split("\n")[1:]))
    df = ps.read_txt(temp_folder+table_temp, verbose=False)

    # calculate correlations
    cor, cov = ps.basis_corr(df, oprint=False)

    # calculate p-values
    abs_cor = np.abs(squareform(cor, checks=False))
    n_sig = np.zeros(abs_cor.shape)
    # TODO: Convert to making bootstraps directly, eliminate read/write
    sparcc.make_bootstraps(df, bootstraps, boot_temp+"#.txt", temp_folder)
    for i in glob.glob(temp_folder+boot_temp+"*.txt"):
        n_sig[np.abs(boostrapped_correlation_lowmem(i)) >= abs_cor] += 1
    p_vals = squareform(1.*n_sig/bootstraps, checks=False)

    # generate correls
    correls = list()
    for i in xrange(len(cor.index)):
        for j in xrange(i+1, len(cor.index)):
            correls.append([str(cor.index[i]), str(cor.index[j]), cor.iat[i, j], p_vals[i, j]])

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust is not None:
        header.append('adjusted_p')

    # cleanup, remove all of bootstraps folder
    shutil.rmtree(temp_folder)

    return correls, header

def paired_correlations_from_table_with_outlier_removal(table, good_samples, min_keep=10, correl_method=spearmanr,
                                                        p_adjust=general.bh_adjust):
    """Takes a biom table and finds correlations between all pairs of otus."""
    correls = list()

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in table.iter_pairwise(axis='observation'):
        samp_union = np.union1d(good_samples[otu_i], good_samples[otu_j])
        # remove zero zero points
        # samp_union = [ind for i, ind in enumerate(samp_union) if data_i[i]!=0 and data_j[i]!=0]
        if len(samp_union) > min_keep:
            correl = correl_method(data_i[samp_union], data_j[samp_union])
            correls.append([str(otu_i), str(otu_j), correl[0], correl[1]])

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust is not None:
        header.append('adjusted_p')

    return correls, header


def make_modules(graph, k=3):
    """make modules with networkx k-clique communities and annotate network
    :type graph: networkx graph
    """
    cliques = list(nx.k_clique_communities(graph, k))
    cliques = [list(i) for i in cliques]
    for i, clique in enumerate(cliques):
        for node in clique:
            graph.node[node]['k_' + str(k)] = i
    return graph, cliques


def make_modules_multik(graph, k=None):
    """make modules with networkx k-clique communities and annotate network"""
    if k is None:
        k = [2, 3, 4, 5, 6]
    communities = dict()
    for k_val in list(k):
        cliques = list(nx.k_clique_communities(graph, k_val))
        cliques = [list(i) for i in cliques]
        communities[k_val] = cliques
        for i, clique in enumerate(cliques):
            for node in clique:
                graph.node[node]['k_' + str(k_val)] = i
    return graph, {k: list(v) for k, v in communities.iteritems()}


def collapse_modules(table, cliques, prefix="module_"):
    """collapse created modules in a biom table"""
    in_module = set()
    modules = np.zeros((len(cliques), table.shape[1]))

    # for each clique merge values and print cliques to file
    with open("cliques.txt", 'w') as f:
        for i, clique in enumerate(cliques):
            in_module = in_module | set(clique)
            f.write(prefix+str(i)+'\t'+','.join([str(j) for j in clique])+'\n')
            for feature in clique:
                try:
                    modules[i] += table.data(feature, axis="observation")
                except UnknownIDError:
                    print feature
                    print feature in table.ids(axis="observation")
                    sys.exit("exit with UnknownIDError")
                except IndexError:
                    print table.data(feature, axis="observation").shape
                    print modules.shape
                    print i
                    print len(cliques)
                    sys.exit("exit with IndexError")
    table.filter(in_module, axis='observation')

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), modules))
    new_table_obs = list(table.ids(axis='observation')) + [prefix + str(i) for i in range(0, len(cliques))]
    return Table(new_table_matrix, new_table_obs, table.ids())


def collapse_modules_multik(table, cliques, prefix="module_"):
    """collapse created modules in a biom table"""
    in_module = set()
    modules = np.zeros((len(cliques), table.shape[1]))

    # for each clique merge values
    for i, clique in enumerate(cliques):
        in_module = in_module | set(clique)
        for feature in clique:
            modules[i] += table.data(feature, axis='observation')
    table.filter(in_module, axis='observation')

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), modules))
    new_table_obs = list(table.ids(axis='observation')) + [prefix + str(i) for i in range(0, len(cliques))]
    return Table(new_table_matrix, new_table_obs, table.ids(), table.metadata(axis="observation"), table.metadata(axis="sample"))


def module_maker(args):
    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'sparcc': sparcc_correlations_multi}
    p_methods = {'bh': general.bh_adjust, 'bon': general.bonferroni_adjust}
    correl_method = correl_methods[args.correl_method]
    if args.p_adjust is not None:
        p_adjust = p_methods[args.p_adjust]
    else:
        p_adjust = None

    # get features to be correlated and extract metadata
    table = load_table(args.input)
    metadata = general.get_metadata_from_table(table)
    print "Table loaded: " + str(table.shape[0]) + " observations"
    print ""

    # make new output directory and change to it
    if args.output is not None:
        os.makedirs(args.output)
        os.chdir(args.output)

    # convert to relative abundance and filter
    if args.min_sample != None:
        table_filt = general.filter_table(table, args.min_sample)
        print "Table filtered: " + str(table_filt.shape[0]) + " observations"
        print ""
    else:
        table_filt = table

    # correlate feature
    if correl_method in [spearmanr, pearsonr]:
        if args.outlier_removal:
            # remove outlier observations
            # first attempt with just looking at individual otu's
            good_samples = general.remove_outliers(table_filt)
            print "Outliers removed: " + str(len(good_samples)) + " observations"
            print ""
            correls, correl_header = paired_correlations_from_table_with_outlier_removal(table_filt, good_samples,
                                                                                         correl_method, p_adjust)
        else:
            # correlate feature
            correls, correl_header = paired_correlations_from_table(table_filt, correl_method, p_adjust)
    else:
        correls, correl_header = sparcc_correlations(table_filt, p_adjust)
    general.print_delimited('correls.txt', correls, correl_header)

    print "Features Correlated"

    # make correlation network
    net = general.correls_to_net(correls, conet=True, metadata=metadata, min_p=args.min_p)
    print "Network Generated"
    print "number of nodes: " + str(net.number_of_nodes())
    print "number of edges: " + str(net.number_of_edges())
    print ""

    # make modules
    net, cliques = make_modules(net)
    print "Modules Formed"
    print "number of modules: " + str(len(cliques))
    print ""

    # print network
    nx.write_gml(net, 'conetwork.gml')

    # collapse modules
    coll_table = collapse_modules(table, cliques, args.prefix)
    print "Table Collapsed"

    # print new table
    coll_table.to_json('make_modules.py', open('collapsed.biom', 'w'))


if __name__ == '__main__':
    """main, takes argparser"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="location of input biom file")
    parser.add_argument("-o", "--output", help="output file location")
    parser.add_argument("-m", "--correl_method", help="correlation method", default="sparcc")
    parser.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    parser.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    parser.add_argument("--prefix", help="prefix for module names in collapsed file", default="module_")
    parser.add_argument("-k", "--k_size", help="desired k-size to determine cliques", default=3, type=int)
    parser.add_argument("--min_p", help="minimum p-value to determine edges", default=.05, type=float)
    parser.add_argument("--outlier_removal", help="outlier detection and removal", default=False, action="store_true")
    module_maker(parser.parse_args())
