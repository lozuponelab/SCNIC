import pytest
from SCNIC.general import simulate_correls, correls_to_net
from SCNIC.correlation_analysis import paired_correlations_from_table
from SCNIC.module_maker import make_modules, collapse_modules, write_modules_to_dir
import os
import glob


@pytest.fixture()
def biom_table1():
    table = simulate_correls()
    return table


@pytest.fixture()
def net1(biom_table1):
    correls = paired_correlations_from_table(biom_table1)
    net = correls_to_net(correls, min_r=.6)
    return net


@pytest.fixture()
def modules1_k3(net1):
    net, modules_k3 = make_modules(net1)
    return modules_k3


def test_module_maker(modules1_k3):
    assert len(modules1_k3) == 1
    assert type(modules1_k3) is dict


def test_collapse_modules(biom_table1, modules1_k3):
    coll_table = collapse_modules(biom_table1, modules1_k3)
    assert coll_table.shape[0] == biom_table1.shape[0]-2
    assert coll_table.shape[1] == biom_table1.shape[1]


def test_write_modules_to_dir(biom_table1, modules1_k3, tmpdir):
    tempdir = tmpdir.mkdir("mods")
    os.chdir(str(tempdir))
    write_modules_to_dir(biom_table1, modules1_k3)
    os.chdir("modules")
    fnames = glob.glob(str(tempdir)+"/modules/*.biom")
    assert len(fnames) == len(modules1_k3)
