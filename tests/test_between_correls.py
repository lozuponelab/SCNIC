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
    """
    Give each test two isolated input biom paths and the same output directory.

    Returns a dictionary of filepaths.
    """
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
    Two tables with very strong, easy-to-detect correlation structures and
    no ambiguity: high sample size, high correlation strength, low noise
    on the correlated features, well-separated means so nothing goes
    negative/degenerate after rounding. Seed is not set here to ensure the tables
    are not the exact same. 

    Returns a dictionary of file paths.
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
                table.to_hdf5(f, f"test_{loc_name.split('_')[0]}")
            new_out_paths[f"{loc_name.split('_')[0]}_table"] = table
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
        #no exception == pass

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
        assert "pAdjusted" in df.columns

    def test_invalid_correl_method_raises(self, strong_signal_table):
        out_paths = strong_signal_table
        with pytest.raises(KeyError):
            between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="not_a_real_method") 

    def test_invalid_pvalue_raises(self, strong_signal_table):
        out_paths = strong_signal_table
        with pytest.raises(Exception) as exc_info:
            between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson", max_p=0.1) 
        assert exc_info.value.args[0] == "SCNIC does not currently support module making based on p-values."

    def test_verbose_flag_does_not_crash(self, strong_signal_table, capsys):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], 
                        correl_method="pearson", verbose=True)   
        captured = capsys.readouterr()
        assert "Correlating" in captured.out or "loaded" in captured.out.lower()


# ---------------------------------------------------------------------------
# Correctness: does it recover the known correlation structure?
# - between doesn't have sparcc and kendalltau correlation options - I wonder why
# ---------------------------------------------------------------------------

class TestCorrelationRecovery:
    """
    These tests check *statistical correctness*, not just "it ran".
    We know the ground truth from simulate_correls, so we assert on it.
    """

    TRIANGLE = {"Observ_0", "Observ_1", "Observ_2"}
    PAIR = {"Observ_3", "Observ_4"}

    @pytest.mark.parametrize("method", ["pearson", "spearman"])
    def test_parametric_methods_recover_triangle_and_pair(self, strong_signal_table, method):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method=method, p_adjust="fdr_bh")   
        net = nx.read_gml(os.path.join(out_paths["output_loc"], f"between_{method}_crossnet.gml"))

        for a, b in itertools.combinations(self.TRIANGLE, 2):
            assert net.has_edge(a, b), f"expected edge {a}-{b} not found ({method})"

        assert net.has_edge(*self.PAIR), f"expected pair edge not found ({method})"

    @pytest.mark.parametrize("method", ["pearson", "spearman"])
    def test_noncorrelated_features_are_not_linked(self, strong_signal_table, method):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method=method, p_adjust="fdr_bh")   
        net = nx.read_gml(os.path.join(out_paths["output_loc"], f"between_{method}_crossnet.gml"))

        noncor_nodes = [n for n in net.nodes if n.startswith("Observ_") and
                         int(n.split("_")[1]) >= 5]

        # since non-significant p-values are not filtered out, need to adjust how we test whether edges are present or not 
        for a, b in itertools.combinations(noncor_nodes, 2):
            if net.has_edge(a,b):
                assert net[a][b].get('pAdjusted') > 0.05
                if AssertionError:
                    f"unexpected significant edge {a}-{b} ({method})"
            else:
                continue

        for a in noncor_nodes:
            for b in self.TRIANGLE | self.PAIR:
                if net.has_edge(a,b):
                    assert net[a][b].get('pAdjusted') > 0.05
                    if AssertionError:
                        f"unexpected cross edge {a}-{b} ({method})"
                else:
                    continue

    ## what does it mean if these tests are failing? - this might be because correlations are not filtered by significance in output
    def test_triangle_and_pair_are_disconnected_from_each_other(self, strong_signal_table):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson", p_adjust="fdr_bh")   
        net = nx.read_gml(os.path.join(out_paths["output_loc"], "between_pearson_crossnet.gml"))
        for a in self.TRIANGLE:
            for b in self.PAIR:
                if net.has_edge(a,b):
                    assert net[a][b].get('pAdjusted') > 0.05
                    if AssertionError:
                        f"unexpected triangle-pair connection at edge {a}-{b}"

                else:
                    continue


    def test_weak_correlation_is_not_falsely_detected(self, tmp_paths):
        """Sanity check the negative case: near-zero correlation strength
        should not produce a densely connected network."""
        out_paths = tmp_paths
        table1 = simulate_correls(corr_stren=(0.0, 0.0), size=50, noncors=10) ## hopefully this generates two different tables...
        with biom_open(out_paths["input1_loc"], "w") as f:
            table1.to_hdf5(f, "input1_test")

        table2 = simulate_correls(corr_stren=(0.0, 0.0), size=50, noncors=10)
        with biom_open(out_paths["input2_loc"], "w") as f:
            table2.to_hdf5(f, "input2_test")
        
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], correl_method="pearson", p_adjust="fdr_bh")  
        net = nx.read_gml(os.path.join(out_paths["output_loc"], "between_pearson_crossnet.gml"))
        n = net.number_of_nodes()
        #print("Number of self-loops:", len(list(nx.selfloop_edges(net)))) - will need this if i decide to keep self-loops for the between analysis!
        num_self_loops = len(list(nx.selfloop_edges(net))) # get number of self loops to add to the max possible edges 
        max_possible_edges = (n * (n - 1)) / 2
        signif_edges = [(a,b) for a, b, c in net.edges(data=True) if c.get('pAdjusted', 0) <= 0.05] # get significant edge pairs based on adjusted p-value - p-value should not be signif if correlation is weak?
        num_signif_edges = len(signif_edges) # get number of significant edge pairs found above
        #print(f"weak correlation table(s) number of total edges: {net.number_of_edges()}")
  
        assert net.number_of_edges() <= (max_possible_edges + num_self_loops) ## changed this to test that edges dont exceed max possible edges
        assert num_signif_edges < 0.1 * (max_possible_edges + num_self_loops) ## now to check that signif edges are less than a fraction of max possible edges


# ---------------------------------------------------------------------------
# Filtering behavior
# ---------------------------------------------------------------------------

class TestFiltering:

    def test_sparcc_filter_reduces_or_maintains_observations(self, strong_signal_table):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], 
                        correl_method="pearson", sparcc_filter=True) 
        df = pd.read_csv(os.path.join(out_paths["output_loc"], "between_pearson_correls.txt"), sep="\t")
        involved = set(df["feature1"]) | set(df["feature2"])
        assert len(involved) <= out_paths["input1_table"].shape[0]
        assert len(involved) <= out_paths["input2_table"].shape[0]

    def test_min_sample_filter_drops_sparse_features(self, tmp_paths):
        out_paths = tmp_paths

        for loc_name,fp in out_paths.items():
            if loc_name in ["input1_loc", "input2_loc"]:
                table = simulate_correls(size=20, noncors=5)
                data = table.matrix_data.toarray()
                data[0, 2:] = 0  # make Observ_0 almost entirely zero across samples
                from biom.table import Table
                sparse_table = Table(data, table.ids("observation"), table.ids("sample"))
                with biom_open(out_paths[loc_name], "w") as f:
                    sparse_table.to_hdf5(f, "test")
        
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], 
                        correl_method="pearson", min_sample=10)
        df = pd.read_csv(os.path.join(out_paths["output_loc"], "between_pearson_correls.txt"), sep="\t")
        involved = set(df["feature1"]) | set(df["feature2"])
        assert "Observ_0" not in involved

    def test_min_sample_and_sparcc_filter_both_set_prefers_sparcc(self, strong_signal_table):
        """When both filters are given, sparcc_filter branch should win
        (per current if/elif ordering) -- document/lock in that behavior."""
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], 
                        correl_method="pearson", sparcc_filter=True, min_sample=5)


# ---------------------------------------------------------------------------
# p-value adjustment
# ---------------------------------------------------------------------------

class TestPAdjust:

    @pytest.mark.parametrize("p_adjust", ["fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky", "bonferroni", "holm"])
    def test_p_adjust_methods_run(self, strong_signal_table, p_adjust):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], 
                        correl_method="pearson", p_adjust=p_adjust)  
        assert os.path.isfile(os.path.join(out_paths["output_loc"], "between_pearson_correls.txt"))

    def test_stricter_adjustment_yields_fewer_or_equal_edges(self, strong_signal_table):
        out_paths = strong_signal_table
        out_bh = out_paths["output_loc"] + "_bh"
        out_bonf = out_paths["output_loc"] + "_bonf"
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_bh, correl_method="pearson", p_adjust="fdr_bh")
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_bonf, correl_method="pearson", p_adjust="bonferroni")
        net_bh = nx.read_gml(os.path.join(out_bh, "between_pearson_crossnet.gml"))
        net_bonf = nx.read_gml(os.path.join(out_bonf, "between_pearson_crossnet.gml"))
        assert net_bonf.number_of_edges() <= net_bh.number_of_edges()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestMultiprocessing:

    def test_results_identical_across_proc_counts(self, strong_signal_table):
        out_paths = strong_signal_table
        out1 = out_paths["output_loc"] + "_1proc"
        out2 = out_paths["output_loc"] + "_2proc"
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out1, correl_method="pearson", procs=1)
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out2, correl_method="pearson", procs=2)
        df1 = pd.read_csv(os.path.join(out1, "between_pearson_correls.txt"), sep="\t")
        df2 = pd.read_csv(os.path.join(out2, "between_pearson_correls.txt"), sep="\t")
        pd.testing.assert_frame_equal(
            df1.sort_values(["feature1", "feature2"]).reset_index(drop=True),
            df2.sort_values(["feature1", "feature2"]).reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_small_tables_two_features(self, tmp_paths):
        out_paths = tmp_paths
        
        for loc_name,fp in out_paths.items():
            if loc_name in ["input1_loc", "input2_loc"]:
                table = simulate_correls(size=15, noncors=0)
                with biom_open(out_paths[loc_name], "w") as f:
                    table.to_hdf5(f, "test")
                
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], 
                                correl_method="pearson")
        assert os.path.isfile(os.path.join(out_paths["output_loc"], "between_pearson_correls.txt"))

    def test_output_dir_already_exists(self, strong_signal_table):
        out_paths = strong_signal_table
        os.makedirs(out_paths["output_loc"])
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], 
                        correl_method="pearson") 

    def test_logger_records_input_and_sample_counts(self, strong_signal_table):
        out_paths = strong_signal_table
        between_correls(out_paths["input1_loc"], out_paths["input2_loc"], out_paths["output_loc"], 
                        correl_method="pearson")
        with open(os.path.join(out_paths["output_loc"], "SCNIC_between_pearson_log.txt")) as f:
            content = f.read()
        assert str(out_paths["input1_table"].shape[0]) and str(out_paths["input2_table"].shape[0]) in content
        assert str(out_paths["input1_table"].shape[1]) and str(out_paths["input2_table"].shape[1]) in content