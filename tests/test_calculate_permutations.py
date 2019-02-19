import pytest

from os import path
import os
from glob import glob
from numpy.testing import assert_almost_equal
import pandas as pd

from SCNIC.calculate_permutations import get_module_sizes_across_rs, perm, run_perms, do_multiprocessed_perms


@pytest.fixture()
def data_loc(tmpdir):
    return tmpdir.mkdir('data')


@pytest.fixture()
def modules():
    return {'module_0': ['otu1', 'otu2', 'otu3'],
            'module_1': ['otu4', 'otu5']}


@pytest.fixture()
def modules_loc(tmpdir, data_loc, modules):
    loc = path.join(data_loc, 'minr_0.35')
    os.mkdir(loc)
    with open(path.join(loc, 'modules.txt'), 'w') as f:
        for module, otus in modules.items():
            f.write('%s\t%s\n' % (module, '\t'.join(otus)))
    return path.join(data_loc, '*', 'modules.txt')


@pytest.fixture()
def modules_across_rs(modules):
    return {'minr_0.35': modules}


@pytest.fixture()
def module_sizes(modules_across_rs):
    return get_module_sizes_across_rs(modules_across_rs)


def test_get_module_sizes_across_rs(module_sizes):
    assert len(module_sizes) == 1
    assert module_sizes['minr_0.35'] == {2, 3}


@pytest.fixture()
def annotated_correls():
    index = [('otu1', 'otu2'),
             ('otu1', 'otu3'),
             ('otu1', 'otu4'),
             ('otu1', 'otu5'),
             ('otu2', 'otu3'),
             ('otu2', 'otu4'),
             ('otu2', 'otu5'),
             ('otu3', 'otu4'),
             ('otu3', 'otu5'),
             ('otu4', 'otu5')]
    columns = ['r', 'PD', 'percent_shared', 'correlated_minr_0.35', 'module_minr_0.35', 'three_plus_minr_0.35',
               'residual_minr_0.35']
    data = [[.9, .001, 2/3,  True, 'module_0',  True, 1.549],
            [.9, .001,   1,  True, 'module_0',  True, 1.88233333334],
            [.1,  .99,   0, False,     'None',  True, -0.106666666663],
            [.1,  .92, 1/5, False,     'None',  True, 0.163333333337],
            [.8,   .8, 2/3,  True, 'module_0',  True, 0.750000000004],
            [.1,  .94,   0, False,     'None',  True, -0.0566666666626],
            [.1,  .96,   0, False,     'None',  True, -0.0766666666626],
            [.1,  .98,   0, False,     'None',  True, -0.0966666666626],
            [.1,  .91, 1/5, False,     'None',  True, 0.173333333337],
            [.9,   .3, 2/3,  True, 'module_1', False, 1.25]]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=columns)


def test_perm(annotated_correls):
    pd_stat, pd_ko_stat = perm(['otu1', 'otu2', 'otu5'], annotated_correls, 'minr_0.35')
    assert_almost_equal(pd_stat, -1.5683439896662443)
    assert_almost_equal(pd_ko_stat, 1.5995043482002014)


@pytest.fixture()
def perms_loc(annotated_correls, module_sizes, data_loc):
    run_perms(annotated_correls, 3, 1, module_sizes, data_loc)
    run_perms(annotated_correls, 3, 1, module_sizes, data_loc)
    return data_loc


def test_run_perms(perms_loc):
    assert len(glob(path.join(perms_loc, 'pd_stats_dict_*'))) == 2
    assert len(glob(path.join(perms_loc, 'pd_ko_stats_dict_*'))) == 2
    pd_stats = pd.read_csv(glob(path.join(perms_loc, 'pd_stats_dict_*'))[0], sep='\t', index_col=(0, 1), header=None)
    assert pd_stats.shape == (1, 3)
    pd_ko_stats = pd.read_csv(glob(path.join(perms_loc, 'pd_ko_stats_dict_*'))[0], sep='\t', index_col=(0, 1),
                              header=None)
    assert pd_ko_stats.shape == (1, 3)


@pytest.fixture()
def correls_anno_loc(annotated_correls, data_loc):
    annotated_correls.to_csv(path.join(data_loc, 'correls_anno.txt'), sep='\t')
    return path.join(data_loc, 'correls_anno.txt')


def test_do_multiprocessed_perms(correls_anno_loc, modules_loc, tmpdir):
    loc = tmpdir.mkdir('test_multi')
    do_multiprocessed_perms(correls_anno_loc, 3, 1, modules_loc, str(loc), None)
    assert len(glob(path.join(loc, 'pd_stats_dict_*.txt'))) == 1
    assert len(glob(path.join(loc, 'pd_ko_stats_dict_*.txt'))) == 1
