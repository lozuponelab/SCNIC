import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import networkx as nx
from biom.table import Table
from biom.util import biom_open

from SCNIC.module import module_maker
from SCNIC import module_analysis as ma

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sim_table():
    arr = np.array([
        [10, 0, 4, 0],
        [10, 0, 4, 0],
        [10, 0, 4, 0],
        [0, 8, 0, 8],
        [0, 8, 0, 8],
    ])
    obs_ids = [f"otu_{i}" for i in range(5)]
    samp_ids = [f"samp_{j}" for j in range(arr.shape[1])]
    obs_meta = [
        {"taxonomy": f"k__Bacteria; p__Firmicutes; g__OTU{i}"}
        for i in range(5)
    ]
    return Table(arr, obs_ids, samp_ids, observation_metadata=obs_meta)


@pytest.fixture()
def sim_correls():
    index = [
        ("otu_0", "otu_1"),
        ("otu_0", "otu_2"),
        ("otu_1", "otu_2"),
        ("otu_3", "otu_4"),
        ("otu_0", "otu_3"),
        ("otu_0", "otu_4"),
        ("otu_1", "otu_3"),
        ("otu_1", "otu_4"),
        ("otu_2", "otu_3"),
        ("otu_2", "otu_4"),
    ]
    r_values = [0.8, 0.8, 0.8, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p_adjusted = [0.01, 0.01, 0.01, 0.01, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
    return pd.DataFrame(
        {"r": r_values, "pAdjusted": p_adjusted},
        index=pd.MultiIndex.from_tuples(index),
    )


@pytest.fixture()
def sim_metadata():
    return {
        f"otu_{i}": {"taxonomy": f"k__Bacteria; p__Firmicutes; g__OTU{i}"}
        for i in range(5)
    }

@pytest.fixture()
def tmp_paths():
    """Give each test an isolated input biom path and output directory."""
    tmpdir = tempfile.mkdtemp()
    input_biom_loc = os.path.join(tmpdir, "input.biom")
    input_correls_loc = os.path.join(tmpdir, "correls.txt")
    output_loc = os.path.join(tmpdir, "output")
    yield input_biom_loc, input_correls_loc, output_loc
    shutil.rmtree(tmpdir, ignore_errors=True)

## do i need sim metadata for this??
@pytest.fixture()
def make_sim_data_files(tmp_paths, sim_table, sim_correls):
    input_biom_loc, input_correls_loc, output_loc = tmp_paths
    ## save biom table to tmp path
    with biom_open(input_biom_loc, "w") as f:
        sim_table.to_hdf5(f, "simulated")
    ## save correls to tmp path
    sim_correls.to_csv(input_correls_loc, sep="\t")
    return input_biom_loc, input_correls_loc, output_loc


# ---------------------------------------------------------------------------
# Basic I/O / smoke tests
# ---------------------------------------------------------------------------

class TestBasicExecution:

    def test_runs_without_error(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        module_maker(input_correls_loc, output_loc, min_r=0.3, method='naive', table_loc=input_biom_loc)
        #no exception == pass
    
    def test_creates_output_directory(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        assert not os.path.isdir(output_loc)
        module_maker(input_correls_loc, output_loc, min_r=0.3, method='naive', table_loc=input_biom_loc)
        assert os.path.isdir(output_loc)

    def test_creates_expected_files(self, make_sim_data_files):
        _, input_correls_loc, output_loc = make_sim_data_files
        module_maker(input_correls_loc, output_loc, min_r=0.3, method='naive')
        assert os.path.isfile(os.path.join(output_loc, "modules.txt"))
        assert os.path.isfile(os.path.join(output_loc, "module_correlation_network.gml"))
        assert os.path.isfile(os.path.join(output_loc, "SCNIC_module_log.txt"))

    def test_creates_collapsed_biom(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        module_maker(input_correls_loc, output_loc, min_r=0.3, method='naive', table_loc=input_biom_loc)
        assert os.path.isfile(os.path.join(output_loc, "collapsed.biom"))

    def test_module_maker_missing_p_column_for_max_p_raises(self, make_sim_data_files, sim_correls):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        correls_no_p = sim_correls.drop(columns=["pAdjusted"])
        correls_no_p.to_csv(input_correls_loc, sep="\t")
        with pytest.raises(ValueError):
            module_maker(input_correls_loc, output_loc, max_p=0.05, method="k_cliques", table_loc=input_biom_loc)

    def test_both_min_r_max_p_raises(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        with pytest.raises(TypeError):
            module_maker(input_correls_loc, output_loc, min_r=0.3, max_p=0.05, table_loc=input_biom_loc)

    def test_no_min_r_max_p_raises(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        with pytest.raises(TypeError):
            module_maker(input_correls_loc, output_loc, table_loc=input_biom_loc)

    def test_invalid_module_method_raises(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        with pytest.raises(KeyError):
            module_maker(input_correls_loc, output_loc, min_r=0.3, method='not_a_real_method', table_loc=input_biom_loc)

    def test_verbose_flag_does_not_crash(self, make_sim_data_files, capsys):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        module_maker(input_correls_loc, output_loc, min_r=0.3, method='naive', table_loc=input_biom_loc, verbose=True)
        captured = capsys.readouterr()
        assert "Modules formed using naive!" and "Network made!" in captured.out

# ---------------------------------------------------------------------------
# Correctness: does each module creation method recover the known structure?
# ---------------------------------------------------------------------------

class TestModuleAnalysis:

    def test_make_modules_naive_returns_expected_groups(self, sim_correls):
        modules = ma.make_modules_naive(sim_correls, min_r=0.5, prefix="module")
        module_sets = {tuple(sorted(m)) for m in modules.values()}
        assert ("otu_0", "otu_1", "otu_2") in module_sets
        assert ("otu_3", "otu_4") in module_sets

    def test_make_modules_k_cliques_returns_expected_modules(self, sim_correls):
        modules = ma.make_modules_k_cliques(sim_correls, min_r=0.5, k=2, prefix="module")
        modules_as_sets = [set(value) for value in modules.values()]
        assert {"otu_0", "otu_1", "otu_2"} in modules_as_sets
        assert {"otu_3", "otu_4"} in modules_as_sets

    def test_make_modules_louvain_returns_two_communities(self, sim_correls):
        modules = ma.make_modules_louvain(sim_correls, min_r=0.5, gamma=1.0, prefix="module")
        modules_as_sets = [set(value) for value in modules.values()]
        assert len(modules_as_sets) == 2
        assert {"otu_0", "otu_1", "otu_2"} in modules_as_sets
        assert {"otu_3", "otu_4"} in modules_as_sets

    def test_make_modules_naive_max_p_raises_not_implemented(self, sim_correls):
        with pytest.raises(NotImplementedError):
            ma.make_modules_naive(sim_correls, max_p=0.05)


# ---------------------------------------------------------------------------
# Filtering behavior
# ---------------------------------------------------------------------------

class TestModuleGML:

    def test_module_gml_respects_min_r(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        module_maker(input_correls_loc, output_loc, min_r=0.5, table_loc=input_biom_loc)

        net = nx.read_gml(os.path.join(output_loc, "module_correlation_network.gml"))
        for _, _, data in net.edges(data=True):
            assert float(data["r"]) >= 0.5

    def test_module_gml_respects_max_p(self, make_sim_data_files):
        _, input_correls_loc, output_loc = make_sim_data_files
        module_maker(input_correls_loc, output_loc, max_p=0.05, method="k_cliques")

        net = nx.read_gml(os.path.join(output_loc, "module_correlation_network.gml"))
        for _, _, data in net.edges(data=True):
            assert float(data["pAdjusted"]) < 0.05

    def test_module_maker_preserves_metadata_in_gml(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        module_maker(input_correls_loc, output_loc, min_r=0.5, table_loc=input_biom_loc)

        net = nx.read_gml(os.path.join(output_loc, "module_correlation_network.gml"))
        assert "taxonomy" in net.nodes["otu_0"]
        assert net.nodes["otu_0"]["taxonomy"] == "k__Bacteria;p__Firmicutes;g__OTU0"


# ---------------------------------------------------------------------------
# Collapse on modules calculations 
# ---------------------------------------------------------------------------

class TestModuleCollapse:

    def test_collapse_modules_sums_rows_correctly(self, sim_table):
        modules = {"module_0": ["otu_0", "otu_1"], "module_1": ["otu_3", "otu_4"]}
        collapsed = ma.collapse_modules(sim_table, modules)

        assert "module_0" in collapsed.ids(axis="observation")
        summed = (
            np.asarray(sim_table.data("otu_0", axis="observation"))
            + np.asarray(sim_table.data("otu_1", axis="observation"))
        )
        module_data = np.asarray(
            collapsed.data("module_0", axis="observation")
        ).ravel()
        assert np.array_equal(module_data, summed)

    def test_collapse_modules_with_invalid_module_label_raises(self, sim_table):
        bad_modules = {"moduleA": ["otu_0", "otu_1"], "module2": ["otu_2", "otu_3"]}
        with pytest.raises(ValueError):
            ma.collapse_modules(sim_table, bad_modules)


# ---------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------- 

class TestModuleEdgeCases:

    def test_empty_correls_returns_empty_modules(self):
        empty = pd.DataFrame(columns=["r"])
        empty.index = pd.MultiIndex.from_tuples([], names=[None, None])

        with pytest.raises(ValueError):
            ma.make_modules_naive(empty, min_r=0.5)

        assert ma.make_modules_k_cliques(empty, min_r=0.5, k=2) == {}
        assert ma.make_modules_louvain(empty, min_r=0.5, gamma=1.0) == {}

    def test_diff_module_prefix(self, make_sim_data_files):
        input_biom_loc, input_correls_loc, output_loc = make_sim_data_files
        module_maker(input_correls_loc, output_loc, min_r=0.3, table_loc=input_biom_loc, prefix='group')

        net = nx.read_gml(os.path.join(output_loc, "module_correlation_network.gml"))
        df = pd.read_csv(os.path.join(output_loc, "modules.txt"), sep="\t")
        assert "group" in net.nodes["otu_0"]["module"]
        assert df.columns.str.contains("group").any()


