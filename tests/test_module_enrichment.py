import pytest
from os import path
import pandas as pd
from biom.table import Table
from scipy.optimize import curve_fit
from numpy.testing import assert_almost_equal
from glob import glob

from SCNIC.module_enrichment import get_modules_across_rs, get_correlation_dicts, percent_shared, annotate_correls,\
    get_module_sizes_across_rs, perm, func, run_perms, get_perms, get_stats, tabulate_stats, do_annotate_correls,\
    do_multiprocessed_perms, do_stats

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
def modules_across_rs(modules_loc):
    return get_modules_across_rs(modules_loc)


def test_get_modules_across_rs(modules_across_rs):
    assert len(modules_across_rs) == 1
    assert len(modules_across_rs[0.35])
    assert len(modules_across_rs[0.35]['module_0']) == 3
    assert len(modules_across_rs[0.35]['module_1']) == 2


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
    columns = ['r', 'PD', 'percent_shared', 'correlated_0.35', 'module_0.35', 'three_plus_0.35']
    data = [[.9, .001, 2/3,  True, 'module_0',  True],
            [.9, .001,   1,  True, 'module_0',  True],
            [.1,  .99,   0, False,     'None',  True],
            [.1,  .92, 1/5, False,     'None',  True],
            [.8,   .8, 2/3,  True, 'module_0',  True],
            [.1,  .94,   0, False,     'None',  True],
            [.1,  .96,   0, False,     'None',  True],
            [.1,  .98,   0, False,     'None',  True],
            [.1,  .91, 1/5, False,     'None',  True],
            [.9,   .3, 2/3,  True, 'module_1', False]]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=columns)


def test_annotate_correls(correls, correls_tip_tips, genome_table, correlation_dicts, annotated_correls):
    correlated_items, modules_membership, module_three_plus = correlation_dicts
    correls_anno = annotate_correls(correls, correls_tip_tips, genome_table, correlated_items, modules_membership,
                                    module_three_plus)
    assert len(correls_anno) == len(correls)
    pd.testing.assert_frame_equal(annotated_correls, correls_anno, check_dtype=False)


######################
@pytest.fixture()
def module_sizes(modules_across_rs):
    return get_module_sizes_across_rs(modules_across_rs)


def test_get_module_sizes_across_rs(module_sizes):
    assert len(module_sizes) == 1
    assert module_sizes[.35] == {2, 3}


def test_perm(annotated_correls):
    non_cor = annotated_correls.loc[annotated_correls['correlated_%s' % 0.35] == False]
    popt_non_cor, _ = curve_fit(func, non_cor.PD.astype(float), non_cor.percent_shared.astype(float))
    non_cor_residuals = non_cor.percent_shared - func(non_cor.PD.astype(float), *popt_non_cor)
    pd_stat, pd_ko_stat = perm(['otu1', 'otu5'], annotated_correls, non_cor.PD, non_cor_residuals,
                               popt_non_cor)
    assert_almost_equal(pd_stat, 1.5)
    assert_almost_equal(pd_ko_stat, 5.5)


@pytest.fixture()
def perms_loc(annotated_correls, module_sizes, modules_loc):
    run_perms(annotated_correls, 3, 1, module_sizes, modules_loc)
    run_perms(annotated_correls, 3, 1, module_sizes, modules_loc)
    return modules_loc


def test_run_perms(perms_loc):
    assert len(glob(path.join(perms_loc, 'pd_stats_dict_*'))) == 2
    assert len(glob(path.join(perms_loc, 'pd_ko_stats_dict_*'))) == 2
    pd_stats = pd.read_table(glob(path.join(perms_loc, 'pd_stats_dict_*'))[0], index_col=(0, 1), header=None)
    assert pd_stats.shape == (1, 3)
    pd_ko_stats = pd.read_table(glob(path.join(perms_loc, 'pd_ko_stats_dict_*'))[0], index_col=(0, 1), header=None)
    assert pd_ko_stats.shape == (1, 3)


@pytest.fixture()
def frames(perms_loc):
    pd_frame = get_perms(path.join(perms_loc, 'pd_stats_dict_*.txt'))
    pd_ko_frame = get_perms(path.join(perms_loc, 'pd_ko_stats_dict_*.txt'))
    return pd_frame, pd_ko_frame


def test_get_perms(frames):
    pd_frame, pd_ko_frame = frames
    assert pd_frame.shape == (1, 6)
    assert pd_ko_frame.shape == (1, 6)


@pytest.fixture()
def stats(annotated_correls, modules_across_rs, frames):
    pd_frame, pd_ko_frame = frames
    stats = get_stats(annotated_correls, modules_across_rs, pd_frame, pd_ko_frame)
    return stats


def test_get_stats(stats):
    assert stats.shape == (1, 7)


def test_tabulate_stats(stats, modules_across_rs):
    tab_stats = tabulate_stats(stats, modules_across_rs)
    assert tab_stats.shape == (1, 5)


########################


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
def correls_anno_loc(annotated_correls, modules_loc):
    annotated_correls.to_csv(path.join(modules_loc, 'correls_anno.txt'), sep='\t')
    return path.join(modules_loc, 'correls_anno.txt')


def test_do_annotate_correls(correls_loc, tree_loc, genome_loc, modules_loc):
    do_annotate_correls(correls_loc, tree_loc, genome_loc, modules_loc, path.join(modules_loc, 'test_correls_anno.txt'))
    assert path.isfile(path.join(modules_loc, 'test_correls_anno.txt'))


def test_do_multiprocessed_perms(correls_anno_loc, modules_loc, tmpdir):
    loc = tmpdir.mkdir('test_multi')
    do_multiprocessed_perms(correls_anno_loc, 3, 1, modules_loc, str(loc))
    assert len(glob(path.join(loc, 'pd_stats_dict_*.txt'))) == 1
    assert len(glob(path.join(loc, 'pd_ko_stats_dict_*.txt'))) == 1


def test_do_stats(correls_anno_loc, modules_loc, perms_loc):
    do_stats(correls_anno_loc, modules_loc, perms_loc, modules_loc)
    assert path.isfile(path.join(modules_loc, 'stats.txt'))
    assert path.isfile(path.join(modules_loc, 'tab_stats.txt'))
    assert path.isfile(path.join(modules_loc, 'pd_sig_plot.png'))
    assert path.isfile(path.join(modules_loc, 'pd_ko_sig_plot.png'))
