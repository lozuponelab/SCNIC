import pytest
from SCNIC.general import simulate_correls
from SCNIC.correlation_analysis import df_to_correls, between_correls_from_tables, calculate_correlations, \
    fastspar_correlation

import pandas as pd
from scipy.stats import pearsonr
from biom import load_table
from os import path
from numpy.testing import assert_allclose
import warnings


@pytest.fixture()
def biom_table1():
    return simulate_correls()


@pytest.fixture()
def data_path():
    return path.join(path.realpath(path.dirname(__file__)), 'data')


@pytest.fixture()
def fastspar_table(data_path):
    return load_table(path.join(data_path, 'fake_data.biom'))


@pytest.fixture()
def correls_spar(data_path):
    correls = pd.read_table(path.join(data_path, 'fake_correls_spar.txt'), index_col=(0, 1), sep='\t',
                            dtype={'feature1': str, 'feature2': str})
    new_index = pd.MultiIndex.from_tuples([(str(i), str(j)) for i, j in correls.index])
    correls.index = new_index
    return correls


# TODO: Induce between table correlations to try to detect
@pytest.fixture()
def biom_table2():
    return simulate_correls()


@pytest.fixture()
def cor():
    labels = ('otu_1', 'otu_2', 'otu_3')
    data = [(1.0, .70, .01),
            (.70, 1.0, .35),
            (.01, .35, 1.0)]
    return pd.DataFrame(data, index=labels, columns=labels)


@pytest.fixture()
def correls():
    index = (('otu_1', 'otu_2'),
             ('otu_1', 'otu_3'),
             ('otu_2', 'otu_3'))
    data = [.7, .01, .35]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=['r'])


def test_df_to_correls(cor, correls):
    cor_correls = df_to_correls(cor, 'r')
    assert cor_correls.equals(correls)


def test_calculate_correlations(biom_table1):
    correls = calculate_correlations(biom_table1, corr_method=pearsonr)
    top_correls = correls.loc[correls.r > .8]
    sig_correls = {('Observ_0', 'Observ_1'), ('Observ_0', 'Observ_2'), ('Observ_1', 'Observ_2'),
                   ('Observ_3', 'Observ_4')}
    assert set(sig_correls) == set(top_correls.index)


def test_fastspar_correlation(fastspar_table, correls_spar):
    correls = fastspar_correlation(fastspar_table)
    assert len(correls.columns) == 1
    assert_allclose(correls.values, correls_spar.values, atol=.1)
    correls_p = fastspar_correlation(fastspar_table, calc_pvalues=True, bootstraps=2)
    assert len(correls_p.columns) == 2


def test_between_correls_from_tables_single(biom_table1, biom_table2):
    correls = between_correls_from_tables(biom_table1, biom_table2)
    assert isinstance(correls, pd.DataFrame)
    assert correls.shape[0] == biom_table1.shape[0] * biom_table2.shape[0]


def test_between_correls_from_tables_multi(biom_table1, biom_table2):
    correls = between_correls_from_tables(biom_table1, biom_table2, nprocs=2)
    assert isinstance(correls, pd.DataFrame)
    assert correls.shape[0] == biom_table1.shape[0] * biom_table2.shape[0]


def test_between_correls_from_tables_too_many_procs(biom_table1, biom_table2):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        correls = between_correls_from_tables(biom_table1, biom_table2, nprocs=1000)
        assert len(w) == 1
    assert isinstance(correls, pd.DataFrame)
    assert correls.shape[0] == biom_table1.shape[0] * biom_table2.shape[0]
