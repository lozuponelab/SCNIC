import os
import shutil
import tempfile
import itertools

import pytest
import networkx as nx
import pandas as pd
import numpy as np
import biom
from biom.util import biom_open

from SCNIC.general import simulate_correls
from SCNIC.between_correls import between_correls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_paths():
    """Give each test an isolated input biom path and output directory."""
    out_paths = {}
    tmpdir = tempfile.mkdtemp()
    out_paths["output_loc"] = os.path.join(tmpdir, "output")

    for num in [1,2]:
        out_paths[f"input{num}_loc"] = os.path.join(tmpdir, f"input{num}.biom")

    yield out_paths
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def strong_signal_table(tmp_paths):
    """
    A table with a very strong, easy-to-detect correlation structure and
    no ambiguity: high sample size, high correlation strength, low noise
    on the correlated features, well-separated means so nothing goes
    negative/degenerate after rounding.
    """
    out_paths = tmp_paths
    new_out_paths = out_paths.copy()
    for loc_name,fp in out_paths.items():
        if loc_name in ["input1_loc", "input2_loc"]:
            ## need to make sure that this doesn't make two tables that are exactly the same
            table = simulate_correls(
                corr_stren=(0.99, 0.99),
                std=(1, 1, 1, 2, 2),
                means=(100, 100, 100, 100, 100),
                size=100,          # plenty of samples for a stable estimate
                noncors=10,
                noncors_mean=100,
                noncors_std=20,    # lower noise than default so it doesn't swamp the signal
            )
            with biom_open(out_paths[loc_name], "w") as f:
                table.to_hdf5(f, f"test_{loc_name.split("_")[0]}")
            new_out_paths[f"{loc_name.split("_")[0]}_table"] = table
        else:
            pass
    return new_out_paths



# ---------------------------------------------------------------------------
# Basic I/O / smoke tests
# ---------------------------------------------------------------------------

class TestBasicExecution:

    def test_runs_without_error(self, strong_signal_table):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson")   
        # no exception == pass

    def test_creates_output_directory(self, strong_signal_table):
        out_paths = strong_signal_table
        assert not os.path.isdir(out_paths["output_loc"])
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson")
        assert os.path.isdir(out_paths["output_loc"])

    def test_creates_expected_files(self, strong_signal_table):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson")   
        assert os.path.isfile(os.path.join(out_paths["output_loc"], "SCNIC_between_pearson_log.txt"))
        assert os.path.isfile(os.path.join(out_paths["output_loc"], "between_pearson_correls.txt"))
        assert os.path.isfile(os.path.join(out_paths["output_loc"], "between_pearson_crossnet.gml"))

    def test_creates_expected_values_minr(self, strong_signal_table):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson", min_r=0.2)
        df = pd.read_csv(os.path.join(out_paths["output_loc"], "between_pearson_correls.txt"), sep="\t")
        assert os.path.isfile(os.path.join(out_paths["output_loc"], "SCNIC_between_pearson_log.txt"))
        assert os.path.isfile(os.path.join(out_paths["output_loc"], "between_pearson_correls.txt"))
        assert os.path.isfile(os.path.join(out_paths["output_loc"], "between_pearson_crossnet.gml"))
        assert not df["r"].any() < 0.2 ## check that r value filtering was successful

    def test_correls_table_has_expected_columns(self, strong_signal_table):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson") 
        df = pd.read_csv(os.path.join(out_paths["output_loc"], "between_pearson_correls.txt"), sep="\t")
        assert "feature1" in df.columns
        assert "feature2" in df.columns
        assert any("r" == c.lower() or "cor" in c.lower() for c in df.columns)
        assert any("p" in c.lower() for c in df.columns)

    def test_invalid_correl_method_raises(self, strong_signal_table):
        out_paths = strong_signal_table
        with pytest.raises(KeyError):
            between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="not_a_real_method") 

    def test_invalid_pvalue_raises(self, strong_signal_table):
        out_paths = strong_signal_table
        with pytest.raises(Exception) as exc_info:
            between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson", max_p=0.1) 
        assert exc_info.value.args[0] == "SCNIC does not currently support module making based on p-values."

