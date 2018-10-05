import pytest
from os import path
import pandas as pd
from biom.table import Table
import numpy as np

from SCNIC.annotate_correls import get_modules_across_rs, get_correlation_dicts, percent_shared, add_correlation_dicts,\
                                   do_annotate_correls, calc_popt, calc_residuals, get_residuals_across_rs,\
                                   add_pd_ko_data


@pytest.fixture()
def modules():
    return {'module_0': ['otu1', 'otu2', 'otu3'],
            'module_1': ['otu4', 'otu5']}


@pytest.fixture()
def modules_loc(tmpdir, modules):
    loc = tmpdir.mkdir('minr_0.35')
    with open(path.join(loc, 'modules.txt'), 'w') as f:
        for module, otus in modules.items():
            f.write('%s\t%s\n' % (module, '\t'.join(otus)))
    return str(loc)


@pytest.fixture()
def modules_across_rs(modules):
    return {0.35: modules}


def test_get_modules_across_rs(modules_across_rs, modules_loc):
    test_modules_across_res = get_modules_across_rs(modules_loc)
    assert len(test_modules_across_res) == 1
    assert len(test_modules_across_res[0.35])
    assert len(test_modules_across_res[0.35]['module_0']) == 3
    assert len(test_modules_across_res[0.35]['module_1']) == 2


@pytest.fixture()
def correls():
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
    rs = [.9,
          .9,
          .1,
          .1,
          .8,
          .1,
          .1,
          .1,
          .1,
          .9]

    return pd.DataFrame([rs], columns=pd.MultiIndex.from_tuples(index), index=['r']).transpose()


@pytest.fixture()
def correlation_dicts(correls, modules_across_rs):
    return get_correlation_dicts(correls, modules_across_rs)


def test_get_correlation_dicts(correlation_dicts):
    correlated_items, modules_membership, module_three_plus = correlation_dicts
    assert len(correlated_items) == 1
    assert sum(correlated_items[0.35]) == 4
    assert len(modules_membership) == 1
    assert modules_membership[0.35] == ['module_0', 'module_0', 'None', 'None', 'module_0',
                                        'None', 'None', 'None', 'None', 'module_1']
    assert len(module_three_plus) == 1
    assert sum(module_three_plus[.35]) == 9


@pytest.fixture()
def genome_frame():
    columns = ['K0001', 'K0002', 'K0003', 'K0004',' K0005']
    index = ['otu1', 'otu2', 'otu3', 'otu4', 'otu5']
    genome_table = [[1, 207, 0, 0, 1],
                    [0,   2, 0, 0, 1],
                    [1,   1, 0, 0, 1],
                    [0,   0, 1, 1, 0],
                    [1,   0, 1, 1, 0]]
    return pd.DataFrame(genome_table, index=index, columns=columns)


@pytest.fixture()
def genome_table(genome_frame):
    return Table(genome_frame.transpose().values, observation_ids=genome_frame.columns, sample_ids=genome_frame.index)


def test_percent_shared(genome_table):
    assert percent_shared(genome_table.data('otu1'), genome_table.data('otu2')) == 2/3
    assert percent_shared(genome_table.data('otu1'), genome_table.data('otu3')) == 1
    assert percent_shared(genome_table.data('otu1'), genome_table.data('otu4')) == 0
    assert percent_shared(genome_table.data('otu1'), genome_table.data('otu5')) == 1/5
    assert percent_shared(genome_table.data('otu4'), genome_table.data('otu5')) == 2/3


@pytest.fixture()
def correls_tip_tips():
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
    pds = [.001,
           .001,
           .99,
           .92,
           .8,
           .94,
           .96,
           .98,
           .91,
           .3]
    return pd.Series(pds, index=pd.MultiIndex.from_tuples(index))


@pytest.fixture()
def correlation_data():
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
    columns = ['correlated_0.35', 'module_0.35', 'three_plus_0.35']
    data = [[True, 'module_0',  True],
            [True, 'module_0',  True],
            [False,     'None',  True],
            [False,     'None',  True],
            [True, 'module_0',  True],
            [False,     'None',  True],
            [False,     'None',  True],
            [False,     'None',  True],
            [False,     'None',  True],
            [True, 'module_1', False]]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=columns)


def test_annotate_correls(correls, correls_tip_tips, genome_table, correlation_dicts, correlation_data):
    correlated_items, modules_membership, module_three_plus = correlation_dicts
    correls_anno = add_correlation_dicts(correls, correlated_items, modules_membership,
                                         module_three_plus)
    assert len(correls_anno) == len(correls)
    pd.testing.assert_frame_equal(correlation_data, correls_anno, check_dtype=False)


@pytest.fixture()
def pd_ko_data():
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
    columns = ['PD', 'percent_shared']
    data = [[.001, 2 / 3],
            [.001, 1],
            [.99, 0],
            [.92, 1 / 5],
            [.8, 2 / 3],
            [.94, 0],
            [.96, 0],
            [.98, 0],
            [.91, 1 / 5],
            [.3, 2 / 3]]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=columns)


def test_add_pd_ko_data(correls, correls_tip_tips, genome_table, pd_ko_data):
    test_pd_ko_data = add_pd_ko_data(correls, correls_tip_tips, genome_table)
    pd.testing.assert_frame_equal(test_pd_ko_data, pd_ko_data, check_dtype=False)


def simple_func(x, a):
    return x + a


@pytest.fixture()
def popt(pd_ko_data):
    return calc_popt(pd_ko_data.PD, pd_ko_data.percent_shared, simple_func)


def test_calc_popt(popt):
    assert len(popt) == 1
    assert popt[0] == -0.3402000000029235


@pytest.fixture()
def residuals():
    return np.array([1.00586667, 1.3392, -0.6498, -0.3798, 0.20686667, -0.5998, -0.6198, -0.6398, -0.3698, 0.70686667])


def test_calc_residuals(pd_ko_data, popt, residuals):
    test_residuals = calc_residuals(pd_ko_data.PD, pd_ko_data.percent_shared, popt, simple_func)
    np.testing.assert_almost_equal(test_residuals, residuals)


@pytest.fixture()
def residual_data(residuals):
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
    columns = ['residuals_0.35']
    return pd.DataFrame(np.transpose(residuals), index=index, columns=columns)


def test_get_residuals_across_rs(correlation_data, pd_ko_data, modules_across_rs, residual_data):
    correls_w_residuals = get_residuals_across_rs(correlation_data, pd_ko_data, modules_across_rs, simple_func)
    assert correls_w_residuals.shape == residual_data.shape


@pytest.fixture()
def correls_loc(modules_loc, correls):
    correls.to_csv(path.join(modules_loc, 'correls.txt'), sep='\t')
    return str(path.join(modules_loc, 'correls.txt'))


@pytest.fixture()
def tree_loc(modules_loc):
    tree = '(otu2:6.0,(otu1:5.0,otu3:3.0,otu4:4.0):5.0,otu5:11.0);'
    with open(path.join(modules_loc, 'tree.nwk'), 'w') as f:
        f.write(tree)
    return str(path.join(modules_loc, 'tree.nwk'))


@pytest.fixture()
def genome_loc(genome_frame, modules_loc):
    genome_frame.to_csv(path.join(modules_loc, 'genome_table.tsv'), sep='\t')
    return str(path.join(modules_loc, 'genome_table.tsv'))


@pytest.fixture()
def annotated_correls(correls, pd_ko_data, residual_data, correlation_data):
    return pd.concat([correls, pd_ko_data, residual_data, correlation_data])


@pytest.fixture()
def correls_anno_loc(annotated_correls, modules_loc):
    annotated_correls.to_csv(path.join(modules_loc, 'correls_anno.txt'), sep='\t')
    return path.join(modules_loc, 'correls_anno.txt')


def test_do_annotate_correls(correls_loc, tree_loc, genome_loc, modules_loc):
    do_annotate_correls(correls_loc, tree_loc, genome_loc, modules_loc, path.join(modules_loc, 'test_correls_anno.txt'),
                        simple_func)
    assert path.isfile(path.join(modules_loc, 'test_correls_anno.txt'))
    test_annotated_correls = pd.read_table(path.join(modules_loc, 'test_correls_anno.txt'), index_col=(0,1))
    assert test_annotated_correls.shape == (10, 7)
