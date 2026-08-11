import os

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

# ---------------------------------------------------------------------------
# Basic I/O / smoke tests
# ---------------------------------------------------------------------------
## TODO:
# - add tests for basic execution similar to test_between_correls.py
# - may need to possibly figure out how to deal with lack of p-value since sparcc doesn't usually return one
# - also max_p isnt supported for module creation? should i just take it out bc it just throws an error when a max_p is specified 

class TestModuleMakerBasicExecution:
    def test_module_maker_with_simulated_data_writes_expected_files(self, tmpdir, sim_correls, sim_table):
        loc = tmpdir.mkdir("module2_test")
        table_path = str(loc.join("table1.biom"))
        correls_path = str(loc.join("correls.txt"))
        out_dir = str(loc.join("out_dir"))

        with biom_open(table_path, "w") as f:
            sim_table.to_hdf5(f, "simulated")

        sim_correls.to_csv(correls_path, sep="\t")

        module_maker(correls_path, out_dir, min_r=0.5, table_loc=table_path)

        files = os.listdir(out_dir)
        assert "modules.txt" in files
        assert "module_correlation_network.gml" in files
        assert "collapsed.biom" in files
        assert "SCNIC_module_log.txt" in files

    def test_module_maker_missing_p_column_for_max_p_raises(self, tmpdir, sim_correls, sim_table):
        loc = tmpdir.mkdir("module2_missing_p")
        table_path = str(loc.join("table1.biom"))
        correls_path = str(loc.join("correls.txt"))
        out_dir = str(loc.join("out_dir"))

        with biom_open(table_path, "w") as f:
            sim_table.to_hdf5(f, "simulated")

        correls_no_p = sim_correls.drop(columns=["pAdjusted"])
        correls_no_p.to_csv(correls_path, sep="\t")

        with pytest.raises(ValueError):
            module_maker(
                correls_path,
                out_dir,
                max_p=0.05,
                method="k_cliques",
                table_loc=table_path,
            )

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
        modules = ma.make_modules_k_cliques(
            sim_correls, min_r=0.5, k=2, prefix="module"
        )
        modules_as_sets = [set(value) for value in modules.values()]
        assert {"otu_0", "otu_1", "otu_2"} in modules_as_sets
        assert {"otu_3", "otu_4"} in modules_as_sets

    def test_make_modules_louvain_returns_two_communities(self, sim_correls):
        modules = ma.make_modules_louvain(
            sim_correls, min_r=0.5, gamma=1.0, prefix="module"
        )
        modules_as_sets = [set(value) for value in modules.values()]
        assert len(modules_as_sets) == 2
        assert {"otu_0", "otu_1", "otu_2"} in modules_as_sets
        assert {"otu_3", "otu_4"} in modules_as_sets

    def test_make_modules_naive_max_p_raises_not_implemented(self, sim_correls):
        with pytest.raises(NotImplementedError):
            ma.make_modules_naive(sim_correls, max_p=0.05)


class TestModuleGML:
    def test_module_gml_respects_min_r(self, tmpdir, sim_correls, sim_table):
        loc = tmpdir.mkdir("module2_gml_minr")
        table_path = str(loc.join("table1.biom"))
        correls_path = str(loc.join("correls.txt"))
        out_dir = str(loc.join("out_dir"))

        with biom_open(table_path, "w") as f:
            sim_table.to_hdf5(f, "simulated")

        sim_correls.to_csv(correls_path, sep="\t")
        module_maker(correls_path, out_dir, min_r=0.5, table_loc=table_path)

        net = nx.read_gml(os.path.join(out_dir, "module_correlation_network.gml"))
        for _, _, data in net.edges(data=True):
            assert float(data["r"]) >= 0.5

    def test_module_gml_respects_max_p(self, tmpdir, sim_correls):
        loc = tmpdir.mkdir("module2_gml_maxp")
        correls_path = str(loc.join("correls.txt"))
        out_dir = str(loc.join("out_dir"))

        sim_correls.to_csv(correls_path, sep="\t")
        module_maker(
            correls_path,
            out_dir,
            max_p=0.05,
            method="k_cliques",
        )

        net = nx.read_gml(os.path.join(out_dir, "module_correlation_network.gml"))
        for _, _, data in net.edges(data=True):
            assert float(data["pAdjusted"]) < 0.05

    def test_module_maker_preserves_metadata_in_gml(self, tmpdir, sim_correls, sim_table):
        loc = tmpdir.mkdir("module2_gml_metadata")
        table_path = str(loc.join("table1.biom"))
        correls_path = str(loc.join("correls.txt"))
        out_dir = str(loc.join("out_dir"))

        with biom_open(table_path, "w") as f:
            sim_table.to_hdf5(f, "simulated")

        sim_correls.to_csv(correls_path, sep="\t")
        module_maker(correls_path, out_dir, min_r=0.5, table_loc=table_path)

        net = nx.read_gml(os.path.join(out_dir, "module_correlation_network.gml"))
        assert "taxonomy" in net.nodes["otu_0"]
        assert net.nodes["otu_0"]["taxonomy"] == "k__Bacteria;p__Firmicutes;g__OTU0"


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
        bad_modules = {"moduleA": ["otu_0", "otu_1"]}
        with pytest.raises(ValueError):
            ma.collapse_modules(sim_table, bad_modules)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
## TODO:
# - expand edge case tests 

class TestModuleEdgeCases:
    def test_empty_correls_returns_empty_modules(self):
        empty = pd.DataFrame(columns=["r"])
        empty.index = pd.MultiIndex.from_tuples([], names=[None, None])

        with pytest.raises(ValueError):
            ma.make_modules_naive(empty, min_r=0.5)

        assert ma.make_modules_k_cliques(empty, min_r=0.5, k=2) == {}
        assert ma.make_modules_louvain(empty, min_r=0.5, gamma=1.0) == {}

