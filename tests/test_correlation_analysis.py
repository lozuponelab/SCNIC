import pytest
from SCNIC.general import simulate_correls
from SCNIC.correlation_analysis import between_correls_from_tables
import pandas as pd
from scipy.misc import comb


@pytest.fixture()
def biom_table1():
    return simulate_correls()


# TODO: Induce between table correlations to try to detect
@pytest.fixture()
def biom_table2():
    return simulate_correls()


def test_between_correls_from_tables(biom_table1, biom_table2):
    correls = between_correls_from_tables(biom_table1, biom_table2)
    assert isinstance(correls, pd.DataFrame)
    assert correls.shape[0] == biom_table1.shape[0] * biom_table2.shape[0]
