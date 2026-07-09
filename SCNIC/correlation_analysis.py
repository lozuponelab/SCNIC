"""
Correlation computation and FastSpar integration.

This module supports pairwise correlation calculation, optional multiprocessing,
and FastSpar-based SparCC-style correlation estimation.
"""

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
    """
    Yield successive n-sized chunks from a list.

    Parameters
    ----------
    l : sequence
        Input sequence.
    n : int
        Chunk size.

    Yields
    ------
    list
        Subsequences of length up to n.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def spearmanr(x, y):
    """
    Wrapper around scipy.stats.spearmanr.

    Parameters
    ----------
    x, y : array-like
        Input vectors for correlation.

    Returns
    -------
    tuple
        Correlation coefficient and p-value.
    """
    return _spearmanr(x, y)


def df_to_correls(cor, col_label='r'):
    """
    Convert a square correlation matrix DataFrame into long-form edge list.

    Parameters
    ----------
    cor : pandas.DataFrame
        Square correlation matrix.
    col_label : str, default 'r'
        Column name for the correlation values in the output.

    Returns
    -------
    pandas.DataFrame
        Long-form dataframe indexed by feature pairs.
    """
    cor.index = [str(i) for i in cor.index]
    cor.columns = [str(i) for i in cor.columns]
    correls = pd.DataFrame(cor.stack().loc[list(combinations(cor.index, 2))], columns=[col_label])
    return correls


def pairwise_iter_wo_metadata(pairwise_iter):
    """
    Strip BIOM pairwise iterator metadata and yield observation pairs.

    Parameters
    ----------
    pairwise_iter : iterator
        Iterator from BIOM table.iter_pairwise().

    Yields
    ------
    tuple
        ((values_i, id_i), (values_j, id_j)) pairs.
    """
    for (val_i, id_i, _), (val_j, id_j, _) in pairwise_iter:
        yield ((val_i, id_i), (val_j, id_j))

def calculate_correlation(data, corr_method=spearmanr):
    """
    Compute correlation for a single feature pair.

    Parameters
    ----------
    data : tuple
        Pair of observation tuples in the form ((values_i, id_i), (values_j, id_j)).
    corr_method : callable, default spearmanr
        Correlation function returning (r, p).

    Returns
    -------
    tuple
        ((id_i, id_j), (r, p)).
    """
    (val_i, id_i), (val_j, id_j) = data
    r, p = corr_method(val_i, val_j)
    return (id_i, id_j), (r, p)


def calculate_correlations(table: Table, corr_method=spearmanr, p_adjust_method: str = 'fdr_bh', nprocs=1) -> pd.DataFrame:
    """
    Calculate pairwise correlations across all observations in a BIOM table.

    Parameters
    ----------
    table : biom.table.Table
        Input BIOM table with observations to correlate.
    corr_method : callable, default spearmanr
        Function used to compute correlation and p-value.
    p_adjust_method : str, default 'fdr_bh'
        Method name for multiple-testing correction.
    nprocs : int, default 1
        Number of processes to use. If 1, runs sequentially.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by feature pairs with columns ``r``, ``p``, and
        optionally ``p_adjusted``.
    """
    if nprocs > multiprocessing.cpu_count():
        warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
        nprocs = multiprocessing.cpu_count()

    ## madi update: bc multiprocessing throwing deadlock/deprecation errors when nproc=1
    ## nprocs <= 1 runs sequentially so no child process is spawned
    ## nprocs > 1 now uses spawn instead of default fork - avoids python 3.12 warning when parent is already multithreaded
    cor = partial(calculate_correlation, corr_method=corr_method)
    if nprocs <= 1:
        results = [cor(item) for item in pairwise_iter_wo_metadata(table.iter_pairwise(axis='observation'))]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(nprocs) as pool:
            results = pool.map(cor, pairwise_iter_wo_metadata(table.iter_pairwise(axis='observation')))
    index = [i[0] for i in results]
    data = [i[1] for i in results]

    correls = pd.DataFrame(data, index=index, columns=['r', 'p'])
    # Turn tuple index into actual multiindex, now guaranteeing that correls index is sorted
    correls.index = pd.MultiIndex.from_tuples([sorted(i) for i in correls.index])
    if p_adjust_method is not None:
        correls['p_adjusted'] = p_adjust(correls.p, method=p_adjust_method)
    return correls


def run_fastspar(otu_table_loc, correl_table_loc, covar_table_loc, stdout=None, nprocs=1):
    """
    Execute the FastSpar binary to compute correlations.

    Parameters
    ----------
    otu_table_loc : str
        Path to the input OTU TSV file.
    correl_table_loc : str
        Path to the output correlation TSV file.
    covar_table_loc : str
        Path to the covariance output TSV file.
    stdout : file-like or None
        Where to direct command stdout.
    nprocs : int, default 1
        Number of FastSpar threads.
    """
    subprocess.run(['fastspar', '-c', otu_table_loc, '-r',correl_table_loc, '-a',
                    covar_table_loc, '-t', str(nprocs)], stdout=stdout, check=True)


def fastspar_correlation(table: Table, verbose: bool=False, calc_pvalues=False, bootstraps=1000, nprocs=1,
                         p_adjust_method='fdr_bh') -> pd.DataFrame:
    """
    Compute FastSpar correlations from a BIOM table.

    Parameters
    ----------
    table : biom.table.Table
        Input BIOM observation table.
    verbose : bool, default False
        If True, stream FastSpar output to the console.
    calc_pvalues : bool, default False
        Whether to calculate bootstrap p-values.
    bootstraps : int, default 1000
        Number of bootstrap replicates.
    nprocs : int, default 1
        Number of threads for FastSpar.
    p_adjust_method : str, default 'fdr_bh'
        Multiple-testing correction method.

    Returns
    -------
    pandas.DataFrame
        Correlation edge list with optional p-values and adjusted p-values.
    """
    with tempfile.TemporaryDirectory(prefix='fastspar') as temp:
        table.to_dataframe(dense=True).to_csv(path.join(temp, 'otu_table.tsv'), sep='\t', index_label='#OTU ID')
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
    """
    Compute correlations between two BIOM tables.

    Parameters
    ----------
    table1 : biom.table.Table
        First input BIOM table.
    table2 : biom.table.Table
        Second input BIOM table.
    correl_method : callable, default spearmanr
        Correlation function.
    nprocs : int, default 1
        Number of worker processes.

    Returns
    -------
    pandas.DataFrame
        DataFrame of pairwise correlations between observations in table1 and table2.
    """
    correls = list()

    if nprocs > multiprocessing.cpu_count():
        warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
        nprocs = multiprocessing.cpu_count()

    ## madi update: bc multiprocessing throwing deadlock/deprecation errors when nproc=1
    ## nprocs <= 1 runs sequentially so no child process is spawned
    ## nprocs > 1 now uses spawn instead of default fork - avoids python 3.12 warning when parent is already multithreaded
    if nprocs <= 1:
        for data_i, otu_i, _ in table1.iter(axis="observation"):
            datas_j = (data_j for data_j, _, _ in table2.iter(axis="observation"))
            corr = partial(correl_method, y=data_i)
            corrs = [corr(data_j) for data_j in datas_j]
            correls += [(otu_i, table2.ids(axis="observation")[i], corrs[i][0], corrs[i][1])
                        for i in range(len(corrs))]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(nprocs) as pool:
            for data_i, otu_i, _ in table1.iter(axis="observation"):
                datas_j = (data_j for data_j, _, _ in table2.iter(axis="observation"))
                corr = partial(correl_method, y=data_i)
                corrs = pool.map(corr, datas_j)
                correls += [(otu_i, table2.ids(axis="observation")[i], corrs[i][0], corrs[i][1])
                            for i in range(len(corrs))]

    correls = pd.DataFrame(correls, columns=['feature1', 'feature2', 'r', 'p'])
    return correls.set_index(['feature1', 'feature2'])  # this needs to be fixed, needs to return multiindex
