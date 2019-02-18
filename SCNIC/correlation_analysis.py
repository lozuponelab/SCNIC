from scipy.stats import spearmanr
import warnings
from functools import partial
import pandas as pd
from biom.table import Table
import subprocess
from itertools import combinations
import tempfile
from os import path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from glob import glob

from SCNIC.general import p_adjust


_spearmanr = spearmanr


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def spearmanr(x, y):
    return _spearmanr(x, y)


def df_to_correls(cor, col_label='r'):
    """takes a square correlation dataframe and turns it into a long form dataframe"""
    cor.index = [str(i) for i in cor.index]
    cor.columns = [str(i) for i in cor.columns]
    correls = pd.DataFrame(cor.stack().loc[list(combinations(cor.index, 2))], columns=[col_label])
    return correls


def pairwise_iter_wo_metadata(pairwise_iter):
    for (val_i, id_i, _), (val_j, id_j, _) in pairwise_iter:
        yield ((val_i, id_i), (val_j, id_j))

def calculate_correlation(data, corr_method=spearmanr):
    (val_i, id_i), (val_j, id_j) = data
    r, p = corr_method(val_i, val_j)
    return (id_i, id_j), (r, p)


def calculate_correlations(table: Table, corr_method=spearmanr, p_adjust_method: str = 'fdr_bh', nprocs=1) -> \
        pd.DataFrame:
    if nprocs > multiprocessing.cpu_count():
        warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
        nprocs = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(nprocs)
    cor = partial(calculate_correlation, corr_method=corr_method)
    results = pool.map(cor, pairwise_iter_wo_metadata(table.iter_pairwise(axis='observation')))
    index = [i[0] for i in results]
    data = [i[1] for i in results]
    pool.close()
    pool.join()
    correls = pd.DataFrame(data, index=index, columns=['r', 'p'])
    # Turn tuple index into actual multiindex, now guaranteeing that correls index is sorted
    correls.index = pd.MultiIndex.from_tuples([sorted(i) for i in correls.index])
    if p_adjust_method is not None:
        correls['p_adjusted'] = p_adjust(correls.p, method=p_adjust_method)
    return correls


def run_fastspar(otu_table_loc, correl_table_loc, covar_table_loc, stdout=None, nprocs=1):
    subprocess.run(['fastspar', '-c', otu_table_loc, '-r',correl_table_loc, '-a',
                    covar_table_loc, '-t', str(nprocs)], stdout=stdout, check=True)


def fastspar_correlation(table: Table, verbose: bool=False, calc_pvalues=False, bootstraps=1000, nprocs=1,
                         p_adjust_method='fdr_bh') -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix='fastspar') as temp:
        table.to_dataframe().to_dense().to_csv(path.join(temp, 'otu_table.tsv'), sep='\t', index_label='#OTU ID')
        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
        run_fastspar(path.join(temp, 'otu_table.tsv'), path.join(temp, path.join(temp, 'correl_table.tsv')),
                     path.join(temp, 'covar_table.tsv'), stdout, nprocs)
        cor = pd.read_csv(path.join(temp, 'correl_table.tsv'), sep='\t', index_col=0)
        correls = df_to_correls(cor)
        if calc_pvalues:
            subprocess.run(['fastspar_bootstrap', '-c', path.join(temp, 'otu_table.tsv'), '-n', str(bootstraps),
                            '-p', path.join(temp, 'boot'), '-t', str(nprocs)], stdout=stdout)
            # infer correlations for each bootstrap count using all available processes
            with ThreadPoolExecutor(max_workers=nprocs) as executor:
                for i in glob((path.join(temp, 'boot*'))):
                    executor.submit(run_fastspar, i, i.replace('boot', 'cor_boot'), i.replace('boot', 'cov_boot'))
            # calculate p_values for correlation table
            subprocess.run(['fastspar_pvalues', '-c', path.join(temp, 'otu_table.tsv'), '-r',
                            path.join(temp, 'correl_table.tsv'), '-p', path.join(temp, 'cor_boot'),
                            '-t', str(nprocs), '-n', str(bootstraps), '-o', path.join(temp, 'pvalues.tsv')],
                           stdout=stdout)
            pvals = pd.read_csv(path.join(temp, 'pvalues.tsv'), sep='\t', index_col=0)
            pvals = df_to_correls(pvals, col_label='p')
            correls = pd.concat([correls, pvals], axis=1, join='inner')
            correls['p_adjusted'] = p_adjust(correls.p, p_adjust_method)
        correls.index = pd.MultiIndex.from_tuples([sorted(i) for i in correls.index])
        return correls


def between_correls_from_tables(table1, table2, correl_method=spearmanr, nprocs=1):
    """Take two biom tables and correlation"""
    correls = list()

    if nprocs > multiprocessing.cpu_count():
        warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
        nprocs = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(nprocs)
    for data_i, otu_i, _ in table1.iter(axis="observation"):
        datas_j = (data_j for data_j, _, _ in table2.iter(axis="observation"))
        corr = partial(correl_method, y=data_i)
        corrs = pool.map(corr, datas_j)
        correls += [(otu_i, table2.ids(axis="observation")[i], corrs[i][0], corrs[i][1])
                    for i in range(len(corrs))]
    pool.close()
    pool.join()

    correls = pd.DataFrame(correls, columns=['feature1', 'feature2', 'r', 'p'])
    return correls.set_index(['feature1', 'feature2'])  # this needs to be fixed, needs to return multiindex
