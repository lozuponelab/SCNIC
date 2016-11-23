import pytest
from SCNIC.general import simulate_correls
from SCNIC.correlation_analysis import paired_correlations_from_table, between_correls_from_tables
import pandas as pd
from scipy.misc import comb


@pytest.fixture()
def biom_table1():
    return simulate_correls()


# TODO: Induce between table correlations to try to detect
@pytest.fixture()
def biom_table2():
    return simulate_correls()


def test_paired_correlations_from_table(biom_table1):
    spearman_correls = paired_correlations_from_table(biom_table1)
    assert type(spearman_correls) is pd.DataFrame
    assert comb(biom_table1.shape[0], 2) == spearman_correls.shape[0]
    pearson_correls = paired_correlations_from_table(biom_table1, correl_method="pearson")
    assert type(pearson_correls) is pd.DataFrame
    assert comb(biom_table1.shape[0], 2) == pearson_correls.shape[0]
    kendall_correls = paired_correlations_from_table(biom_table1, correl_method="kendall")
    assert type(kendall_correls) is pd.DataFrame
    assert comb(biom_table1.shape[0], 2) == kendall_correls.shape[0]


def test_between_correls_from_tables(biom_table1, biom_table2):
    correls = between_correls_from_tables(biom_table1, biom_table2)
    assert type(correls) is pd.DataFrame
    assert correls.shape[0] == biom_table1.shape[0] * biom_table2.shape[0]
