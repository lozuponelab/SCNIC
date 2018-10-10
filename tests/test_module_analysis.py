import pytest
from SCNIC.module_analysis import correls_to_cor, make_modules_naive, collapse_modules, write_modules_to_dir, cor_to_dist, \
                                  write_modules_to_file, add_modules_to_metadata
import os
import glob
from biom.table import Table
import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
from pandas.testing import assert_frame_equal


@pytest.fixture()
def correls():
    index = (('4b5eeb300368260019c1fbc7a3c718fc', 'fe30ff0f71a38a39cf1717ec2be3a2fc'),
             ('4b5eeb300368260019c1fbc7a3c718fc', '154709e160e8cada6bfb21115acc80f5'),
             ('fe30ff0f71a38a39cf1717ec2be3a2fc', '154709e160e8cada6bfb21115acc80f5'))
    data = [.7, .01, .35]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=['r'])


@pytest.fixture()
def cor():
    data = [(1.0, .70, .01),
            (.70, 1.0, .35),
            (.01, .35, 1.0)]
    ids = ['4b5eeb300368260019c1fbc7a3c718fc', 'fe30ff0f71a38a39cf1717ec2be3a2fc', '154709e160e8cada6bfb21115acc80f5']
    return squareform(np.array(data), checks=False), ids


@pytest.fixture()
def dist():
    data = [(0, .15, .495),
            (.15, 0, .325),
            (.495, .325, 0)]
    return squareform(np.array(data), checks=False)


@pytest.fixture()
def biom_table1():
    arr = np.array([[250,   0, 100, 446,   75],
                    [  0,   0,   1,   1,    2],
                    [  2,   2,   2,   2,    2],
                    [100, 100, 500,   1, 1000],
                    [500,   5,   0,  50,  100]])
    obs_ids = ["otu_%s" % i for i in range(5)]
    samp_ids = ["samp_%s" % i for i in range(5)]
    return Table(arr, obs_ids, samp_ids)


@pytest.fixture()
def dist1():
    dist = np.array([[0.0, .05, 0.2, 1.0, 1.0],
                     [.05, 0.0, 0.2, 1.0, 1.0],
                     [0.2, 0.2, 0.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 0.0, 0.7],
                     [1.0, 1.0, 1.0, 0.7, 0.0]])
    return squareform(dist)


@pytest.fixture()
def obs_ids1():
    obs_ids = ["otu_%s" % i for i in range(5)]
    return obs_ids


@pytest.fixture()
def modules1(correls, obs_ids1):
    return make_modules_naive(correls, min_r=.6)


@pytest.fixture()
def modules2():
    return {'module_0': ['otu_0', 'otu_1', 'otu_2'], 'module_1': ['otu_3', 'otu_4']}


@pytest.fixture()
def metadata():
    meta = {
        '4b5eeb300368260019c1fbc7a3c718fc': {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Staphylococcaceae; g__Staphylococcus; s__'},
        'fe30ff0f71a38a39cf1717ec2be3a2fc': {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Paenibacillaceae; g__Paenibacillus; s__'},
        '154709e160e8cada6bfb21115acc80f5': {'taxonomy': 'k__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Methylophilales; f__Methylophilaceae; g__; s__'},
    }
    return meta


def test_correls_to_cor(correls, cor):
    test_cor, test_labels = correls_to_cor(correls)
    test_cor_square = pd.DataFrame(squareform(test_cor), index=test_labels, columns=test_labels)
    test_cor_square = test_cor_square.sort_index(axis=0).sort_index(axis=1)
    cor_square = pd.DataFrame(squareform(cor[0]), index=cor[1], columns=cor[1])
    cor_square = cor_square.sort_index(axis=0).sort_index(axis=1)
    assert_frame_equal(test_cor_square, cor_square)


def test_cor_to_dist(cor, dist):
    test_dist = cor_to_dist(cor[0])
    assert np.allclose(test_dist, dist)


def test_make_modules(modules2):
    assert type(modules2) == dict
    assert len(modules2) == 2
    assert np.sum([len(otus) for module_, otus in modules2.items()]) == 5


def test_collapse_modules(biom_table1, modules2):
    coll_table = collapse_modules(biom_table1, modules2)
    assert coll_table.shape[0] == biom_table1.shape[0]-3
    assert coll_table.shape[1] == biom_table1.shape[1]
    assert coll_table.sum() == biom_table1.sum()
    assert np.array_equal(coll_table.sum(axis="sample"), biom_table1.sum(axis="sample"))
    # set since order doesn't matter
    assert frozenset(coll_table.sum(axis="observation")) == frozenset(np.array([885, 2356]))


def test_write_modules_to_dir(biom_table1, modules2, tmpdir):
    tempdir = tmpdir.mkdir("mods")
    os.chdir(str(tempdir))
    write_modules_to_dir(biom_table1, modules2)
    os.chdir("modules")
    fnames = glob.glob(str(tempdir)+"/modules/*.biom")
    assert len(fnames) == len(modules2)


def test_write_modules_to_file(modules1, tmpdir):
    path = tmpdir.join('modules.txt')
    write_modules_to_file(modules1, path_str=str(path))
    data = open(str(path)).readlines()
    assert len(data) == 1
    assert len(data[0].strip().split()) == 3


def test_add_module_to_metadata(modules1, metadata):
    test_metadata = add_modules_to_metadata(modules1, metadata)
    assert len(metadata) == len(test_metadata)
    assert len(test_metadata['4b5eeb300368260019c1fbc7a3c718fc']) == 2
    assert test_metadata['4b5eeb300368260019c1fbc7a3c718fc']['module'] == 'module_0'
    assert len(test_metadata['fe30ff0f71a38a39cf1717ec2be3a2fc']) == 2
    assert test_metadata['fe30ff0f71a38a39cf1717ec2be3a2fc']['module'] == 'module_0'
    assert len(test_metadata['154709e160e8cada6bfb21115acc80f5']) == 1
