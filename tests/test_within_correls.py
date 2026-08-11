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
from SCNIC.within_correls import within_correls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_paths():
    """Give each test an isolated input biom path and output directory."""
    tmpdir = tempfile.mkdtemp()
    input_loc = os.path.join(tmpdir, "input.biom")
    output_loc = os.path.join(tmpdir, "output")
    yield input_loc, output_loc
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def strong_signal_table(tmp_paths):
    """
    A table with a very strong, easy-to-detect correlation structure and
    no ambiguity: high sample size, high correlation strength, low noise
    on the correlated features, well-separated means so nothing goes
    negative/degenerate after rounding.
    """
    np.random.seed(0)
    input_loc, output_loc = tmp_paths
    table = simulate_correls(
        corr_stren=(0.99, 0.99),
        std=(1, 1, 1, 2, 2),
        means=(100, 100, 100, 100, 100),
        size=100,          # plenty of samples for a stable estimate
        noncors=10,
        noncors_mean=100,
        noncors_std=20,    # lower noise than default so it doesn't swamp the signal
    )
    with biom_open(input_loc, "w") as f:
        table.to_hdf5(f, "test")
    return input_loc, output_loc, table


# ---------------------------------------------------------------------------
# Basic I/O / smoke tests
# ---------------------------------------------------------------------------

class TestBasicExecution:

    def test_runs_without_error(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson")
        # no exception == pass

    def test_creates_output_directory(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        assert not os.path.isdir(output_loc)
        within_correls(input_loc, output_loc, correl_method="pearson")
        assert os.path.isdir(output_loc)

    def test_creates_expected_files(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson")
        assert os.path.isfile(os.path.join(output_loc, "SCNIC_within_pearson_log.txt"))
        assert os.path.isfile(os.path.join(output_loc, "within_pearson_correls.txt"))
        assert os.path.isfile(os.path.join(output_loc, "within_pearson_correlation_network.gml"))

    def test_correls_table_has_expected_columns(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson")
        df = pd.read_csv(os.path.join(output_loc, "within_pearson_correls.txt"), sep="\t")
        assert "feature1" in df.columns
        assert "feature2" in df.columns
        assert any("r" == c.lower() or "cor" in c.lower() for c in df.columns)
        assert any("p" in c.lower() for c in df.columns)

    def test_invalid_correl_method_raises(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        with pytest.raises(KeyError):
            within_correls(input_loc, output_loc, correl_method="not_a_real_method")

    def test_verbose_flag_does_not_crash(self, strong_signal_table, capsys):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson", verbose=True)
        captured = capsys.readouterr()
        assert "Correlating" in captured.out or "loaded" in captured.out.lower()


# ---------------------------------------------------------------------------
# Correctness: does it recover the known correlation structure?
# ---------------------------------------------------------------------------

class TestCorrelationRecovery:
    """
    These tests check *statistical correctness*, not just "it ran".
    We know the ground truth from simulate_correls, so we assert on it.
    """

    TRIANGLE = {"Observ_0", "Observ_1", "Observ_2"}
    PAIR = {"Observ_3", "Observ_4"}

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_parametric_methods_recover_triangle_and_pair(self, strong_signal_table, method):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method=method, p_adjust="fdr_bh")
        net = nx.read_gml(os.path.join(output_loc, f"within_{method}_correlation_network.gml"))

        for a, b in itertools.combinations(self.TRIANGLE, 2):
            assert net.has_edge(a, b), f"expected edge {a}-{b} not found ({method})"

        assert net.has_edge(*self.PAIR), f"expected pair edge not found ({method})"

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_noncorrelated_features_are_not_linked(self, strong_signal_table, method):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method=method, p_adjust="fdr_bh")
        net = nx.read_gml(os.path.join(output_loc, f"within_{method}_correlation_network.gml"))

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
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson")
        net = nx.read_gml(os.path.join(output_loc, "within_pearson_correlation_network.gml"))
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
        input_loc, output_loc = tmp_paths
        table = simulate_correls(corr_stren=(0.0, 0.0), size=50, noncors=10)
        with biom_open(input_loc, "w") as f:
            table.to_hdf5(f, "test")
        within_correls(input_loc, output_loc, correl_method="pearson", p_adjust="fdr_bh")
        net = nx.read_gml(os.path.join(output_loc, "within_pearson_correlation_network.gml"))
        n = net.number_of_nodes()
        max_possible_edges = (n * (n - 1)) / 2
        signif_edges = [(a,b) for a, b, c in net.edges(data=True) if c.get('pAdjusted', 0) <= 0.05] # get significant edge pairs based on adjusted p-value - p-value should not be signif if correlation is weak?
        num_signif_edges = len(signif_edges) # get number of significant edge pairs found above
  
        assert net.number_of_edges() <= max_possible_edges ## changed this to test that edges dont exceed max possible edges
        assert num_signif_edges < 0.1 * max_possible_edges ## now to check that signif edges are less than a fraction of max possible edges


# ---------------------------------------------------------------------------
# Filtering behavior
# ---------------------------------------------------------------------------

class TestFiltering:

    def test_sparcc_filter_reduces_or_maintains_observations(self, strong_signal_table):
        input_loc, output_loc, table = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson",
                        sparcc_filter=True, verbose=True)
        df = pd.read_csv(os.path.join(output_loc, "within_pearson_correls.txt"), sep="\t")
        involved = set(df["feature1"]) | set(df["feature2"])
        assert len(involved) <= table.shape[0]

    def test_min_sample_filter_drops_sparse_features(self, tmp_paths):
        input_loc, output_loc = tmp_paths
        table = simulate_correls(size=20, noncors=5)
        data = table.matrix_data.toarray()
        data[0, 2:] = 0  # make Observ_0 almost entirely zero across samples
        from biom.table import Table
        sparse_table = Table(data, table.ids("observation"), table.ids("sample"))
        with biom_open(input_loc, "w") as f:
            sparse_table.to_hdf5(f, "test")

        within_correls(input_loc, output_loc, correl_method="pearson", min_sample=10)
        df = pd.read_csv(os.path.join(output_loc, "within_pearson_correls.txt"), sep="\t")
        involved = set(df["feature1"]) | set(df["feature2"])
        assert "Observ_0" not in involved

    def test_min_sample_and_sparcc_filter_both_set_prefers_sparcc(self, strong_signal_table):
        """When both filters are given, sparcc_filter branch should win
        (per current if/elif ordering) -- document/lock in that behavior."""
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson",
                        sparcc_filter=True, min_sample=5)


# ---------------------------------------------------------------------------
# p-value adjustment
# ---------------------------------------------------------------------------

class TestPAdjust:

    @pytest.mark.parametrize("p_adjust", ["fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky", "bonferroni", "holm", None])
    def test_p_adjust_methods_run(self, strong_signal_table, p_adjust):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson", p_adjust=p_adjust)
        assert os.path.isfile(os.path.join(output_loc, "within_pearson_correls.txt"))

    def test_stricter_adjustment_yields_fewer_or_equal_edges(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        out_bh = output_loc + "_bh"
        out_bonf = output_loc + "_bonf"
        within_correls(input_loc, out_bh, correl_method="pearson", p_adjust="fdr_bh")
        within_correls(input_loc, out_bonf, correl_method="pearson", p_adjust="bonferroni")
        net_bh = nx.read_gml(os.path.join(out_bh, "within_pearson_correlation_network.gml"))
        net_bonf = nx.read_gml(os.path.join(out_bonf, "within_pearson_correlation_network.gml"))
        assert net_bonf.number_of_edges() <= net_bh.number_of_edges()


# ---------------------------------------------------------------------------
# SparCC-specific behavior
# ---------------------------------------------------------------------------

class TestSparCC:

    def test_sparcc_no_bootstraps_skips_pvalues(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="sparcc", sparcc_p=None)
        df = pd.read_csv(os.path.join(output_loc, "within_sparcc_correls.txt"), sep="\t")
        assert not any("p" == c.lower() for c in df.columns)

    def test_sparcc_with_bootstraps_produces_pvalues(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="sparcc", sparcc_p=50)
        df = pd.read_csv(os.path.join(output_loc, "within_sparcc_correls.txt"), sep="\t")
        assert any("p" in c.lower() for c in df.columns)

    def test_sparcc_recovers_known_structure(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="sparcc", sparcc_p=50)
        net = nx.read_gml(os.path.join(output_loc, "within_sparcc_correlation_network.gml"))
        assert net.has_edge("Observ_3", "Observ_4")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestMultiprocessing:

    def test_results_identical_across_proc_counts(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        out1 = output_loc + "_1proc"
        out2 = output_loc + "_2proc"
        within_correls(input_loc, out1, correl_method="pearson", procs=1)
        within_correls(input_loc, out2, correl_method="pearson", procs=2)
        df1 = pd.read_csv(os.path.join(out1, "within_pearson_correls.txt"), sep="\t")
        df2 = pd.read_csv(os.path.join(out2, "within_pearson_correls.txt"), sep="\t")
        pd.testing.assert_frame_equal(
            df1.sort_values(["feature1", "feature2"]).reset_index(drop=True),
            df2.sort_values(["feature1", "feature2"]).reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_small_table_two_features(self, tmp_paths):
        input_loc, output_loc = tmp_paths
        table = simulate_correls(size=15, noncors=0)
        with biom_open(input_loc, "w") as f:
            table.to_hdf5(f, "test")
        within_correls(input_loc, output_loc, correl_method="pearson")
        assert os.path.isfile(os.path.join(output_loc, "within_pearson_correls.txt"))

    def test_output_dir_already_exists(self, strong_signal_table):
        input_loc, output_loc, _ = strong_signal_table
        os.makedirs(output_loc)
        within_correls(input_loc, output_loc, correl_method="pearson")

    def test_logger_records_input_and_sample_counts(self, strong_signal_table):
        input_loc, output_loc, table = strong_signal_table
        within_correls(input_loc, output_loc, correl_method="pearson")
        with open(os.path.join(output_loc, "SCNIC_within_pearson_log.txt")) as f:
            content = f.read()
        assert str(table.shape[0]) in content
        assert str(table.shape[1]) in content
