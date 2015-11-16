import shutil
import glob
import os

import general
import pysurvey as ps
from pysurvey import SparCC as sparcc
from functools import partial
from scipy.spatial.distance import squareform
import numpy as np

__author__ = 'shafferm'


def boostrapped_correlation_multi(tup, temp_folder, cor_temp):
    """tup is a tuple where tup[0] is num and tup[1] is the file name"""
    df = ps.read_txt(tup[1], verbose=False)
    cor = ps.basis_corr(df, oprint=False)[0]
    ps.write_txt(cor, temp_folder+cor_temp+str(tup[0])+".txt", T=False)


def sparcc_correlations_multi(table, p_adjust=general.bh_adjust, temp_folder=os.getcwd()+"/temp/",
                              boot_temp="bootstrap_", cor_temp="cor_", table_temp="temp_table.txt",
                              bootstraps=100, procs=None):
    """Calculate correlations with sparcc"""
    import multiprocessing

    os.mkdir(temp_folder)
    if procs == None:
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    else:
        pool = multiprocessing.Pool(procs)

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
    for i, _file in enumerate(glob.glob(temp_folder+boot_temp+"*.txt")):
        boostrapped_correlation(_file, temp_folder, cor_temp, i)

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
                               boot_temp="bootstrap_", table_temp="temp_table.txt", bootstraps=100):
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


def boostrapped_correlation_lowmem_multi(in_file, cor):
    df = ps.read_txt(in_file, verbose=False)
    in_cor = ps.basis_corr(df, oprint=False)[0]
    in_cor = squareform(in_cor, checks=False)
    return np.abs(in_cor) >= cor


def sparcc_correlations_lowmem_multi(table, p_adjust=general.bh_adjust, temp_folder=os.getcwd()+"/temp/",
                                     boot_temp="bootstrap_", table_temp="temp_table.txt",
                                     bootstraps=100, procs=None):
    """"""
    # setup
    import multiprocessing

    os.mkdir(temp_folder)
    if procs is None:
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    else:
        pool = multiprocessing.Pool(procs)

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
    pfun = partial(boostrapped_correlation_lowmem_multi, cor=abs_cor)
    multi_results = pool.map(pfun, glob.glob(temp_folder+boot_temp+"*.txt"))
    pool.close()
    pool.join()

    for i in multi_results:
        n_sig[i] += 1
    p_vals = squareform(1.*n_sig/bootstraps, checks=False)

    # generate correls array
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
