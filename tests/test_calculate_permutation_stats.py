import pytest

from os import path
from SCNIC.calculate_permutations import run_perms, get_module_sizes_across_rs
import pandas as pd

from SCNIC.calculate_permutation_stats import get_perms, get_stats, tabulate_stats, do_stats


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


@pytest.fixture()
def module_sizes(modules_across_rs):
    return get_module_sizes_across_rs(modules_across_rs)


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
    columns = ['r', 'PD', 'percent_shared', 'correlated_0.35', 'module_0.35', 'three_plus_0.35', 'residual_0.35']
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


@pytest.fixture()
def perms_loc(annotated_correls, module_sizes, modules_loc):
    run_perms(annotated_correls, 3, 1, module_sizes, modules_loc)
    run_perms(annotated_correls, 3, 1, module_sizes, modules_loc)
    return modules_loc


@pytest.fixture()
def frames(perms_loc):
    pd_frame = get_perms(path.join(perms_loc, 'pd_stats_dict_*.txt'))
    pd_ko_frame = get_perms(path.join(perms_loc, 'pd_ko_stats_dict_*.txt'))
    return pd_frame, pd_ko_frame


@pytest.fixture()
def correls_anno_loc(annotated_correls, modules_loc):
    annotated_correls.to_csv(path.join(modules_loc, 'correls_anno.txt'), sep='\t')
    return path.join(modules_loc, 'correls_anno.txt')


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


def test_do_stats(correls_anno_loc, modules_loc, perms_loc):
    do_stats(correls_anno_loc, modules_loc, perms_loc, modules_loc)
    assert path.isfile(path.join(modules_loc, 'stats.txt'))
    assert path.isfile(path.join(modules_loc, 'tab_stats.txt'))
    assert path.isfile(path.join(modules_loc, 'pd_sig_plot.png'))
    assert path.isfile(path.join(modules_loc, 'pd_ko_sig_plot.png'))
    assert path.isfile(path.join(modules_loc, 'pd_pvalue_boxplots.png'))
    assert path.isfile(path.join(modules_loc, 'pd_ko_pvalue_boxplots.png'))
