#!/usr/local/bin/python3

"""
SCNIC command-line wrapper (called using scnic-analysis).

This script exposes three subcommands for SCNIC workflows:
- within   : find pairwise correlations within a single BIOM table
- modules  : build modules from a correlation network
- between  : find pairwise correlations between two BIOM tables 

Each subcommand accepts its own CLI options documented below.
"""

import argparse
from SCNIC.within_correls import within_correls
from SCNIC.between_correls import between_correls
from SCNIC.module import module_maker

## pull version from git tag and setuptools_scm 
## if this doesn't work, set the version to "0.0.0"
try:
    from SCNIC._version import version as __version__
except ModuleNotFoundError:
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root="..", relative_to=__file__)
    except Exception:
        __version__ = "0.0.0"

__author__ = 'shafferm'

def build_parser():
    """
    Parse command-line arguments and dispatch SCNIC subcommands.

    Subcommands
    -----------
    within
        -i/--input_loc : location of the input BIOM file
        -o/--output_loc : output directory for the analysis results
        -m/--correl_method : correlation method to use
        -a/--p_adjust : p-value adjustment method
        -s/--min_sample : minimum number of samples required for inclusion
        --procs : number of processors to use
        --sparcc_filter : apply the SparCC filter described in the SparCC paper
        --sparcc_p : number of bootstraps for SparCC p-value estimation
        -v/--verbose : print verbose progress messages

    modules
        -i/--input_loc : location of the correlation output from the within step
        -o/--output_loc : output directory for module results
        --max_p : maximum p-value allowed for retaining edges
        --min_r : minimum correlation coefficient required for retaining edges
        --method : module detection method to use
        -k/--k_size : size of the k-clique used when k_cliques is selected
        -g/--gamma : gamma parameter for louvain modularity optimization
        --table_loc : BIOM table used to collapse the network into modules
        --prefix : prefix for module names in the collapsed output
        -v/--verbose : print verbose progress messages

    between
        -1/--table1 : first BIOM table to correlate
        -2/--table2 : second BIOM table to correlate
        -o/--output_loc : output file or directory location
        -m/--correl_method : correlation method to use
        -a/--p_adjust : p-value adjustment method
        -s/--min_sample : minimum number of samples required for inclusion
        --max_p : maximum p-value allowed for retaining edges
        --min_r : minimum correlation coefficient required for retaining edges
        --sparcc_filter : apply the SparCC filter described in the SparCC paper
        --procs : number of processors to use
        -f/--force : overwrite an existing output directory if present
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--version",
                        action="version",
                        version=f"{__version__}")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    subparsers.metavar = "{within, modules, between}" ## give helpful error message when no positional argument is given after "scnic"

    within_corr = subparsers.add_parser("within", 
                                        help="find pairwise correlations within a table",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    make_modules = subparsers.add_parser("modules", 
                                         help="make modules on a network",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    between_corr = subparsers.add_parser("between", 
                                         help="find correlations between two tables",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser for making correlation network with a single biom table
    within_corr.add_argument("-i", "--input_loc", 
                             help="location of input BIOM file", 
                             required=True)
    within_corr.add_argument("-o", "--output_loc", 
                             help="location and desired name of output directory",
                             default="scnic_within_out")
    within_corr.add_argument("-m", "--correl_method", 
                             help="correlation method", 
                             default="sparcc",
                             choices=["sparcc", "spearman", "pearson", "kendall"])
    within_corr.add_argument("-a", "--p_adjust", 
                             help="p-value adjustment, default is Benjamini-Hochberg FDR", 
                             default="fdr_bh",
                             choices=["fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky", "bonferroni", "holm"]) ## include the fwer (family-wise error rate) options - bonferroni, sidak, holm, holm-sidak, simes-hochberg, hommel?
    within_corr.add_argument("-s", "--min_sample", 
                             help="minimum number of samples present in", 
                             type=int)
    within_corr.add_argument("--procs", 
                             help="number of processors to use", 
                             default=1, 
                             type=int)
    within_corr.add_argument("--sparcc_filter", 
                             help="filter as described in SparCC paper", 
                             default=False,
                             action="store_true")
    within_corr.add_argument("--sparcc_p", 
                             help="calculate p-value for sparCC R value, give number of bootstraps",
                             type=int)
    within_corr.add_argument("-v", "--verbose", 
                             help="give verbose messages to STDOUT", 
                             default=False, 
                             action="store_true")
    within_corr.set_defaults(func=within_correls)

    # parser for finding modules in a correlation network
    make_modules.add_argument("-i", "--input_loc", 
                              help="location of output from 'scnic within' or 'scnic-analysis within'. will be scnic_within_out/ if default was used.",
                              required=True)
    make_modules.add_argument("-o", "--output_loc", 
                              help="location and desired name of the output directory",
                              default="scnic_make_modules_out")
    make_modules.add_argument("--max_p", 
                              help="maximum p-value to determine edges, p must have been calculated",
                              type=float)
    make_modules.add_argument("--min_r", 
                              help="minimum correlation value to determine edges", 
                              type=float)
    make_modules.add_argument("--method", 
                              help="method to be used for determining modules", 
                              default='naive',
                              choices=["naive", "k_cliques", "louvain"])
    make_modules.add_argument("-k", "--k_size", 
                              help="k value for use with the k-clique communities algorithm",
                              type=int, 
                              default=3)
    make_modules.add_argument("-g", "--gamma", 
                              help="gamma value for use with louvain modularity maximization, between "
                                   "0 and 1 where 0 makes small modules and 1 large modules",
                              type=float, 
                              default=0.1)
    make_modules.add_argument("--table_loc", 
                              help="biom table used to make network to be collapsed")
    make_modules.add_argument("--prefix", 
                              help="prefix for module names in collapsed file", 
                              default="module")
    make_modules.add_argument("-v", "--verbose", 
                              help="give verbose messages to STDOUT", 
                              default=False,
                              action="store_true")
    make_modules.set_defaults(func=module_maker)

    # parser for building a bipartite correlation network between two data types
    between_corr.add_argument("-1", "--table1", 
                              help="location of first BIOM table to be correlated", 
                              required=True)
    between_corr.add_argument("-2", "--table2", 
                              help="location of second BIOM table to be correlated", 
                              required=True)
    between_corr.add_argument("-o", "--output_loc", 
                              help="location and desired name of output directory",
                              default="scnic_between_out")
    between_corr.add_argument("-m", "--correl_method", 
                              help="correlation method", 
                              default="spearman",
                              choices=["spearman", "pearson"]) ## i dont think there's an option for sparcc here - why?
    between_corr.add_argument("-a", "--p_adjust", 
                              help="p-value adjustment, default is Benjamini-Hochberg FDR", 
                              default="fdr_bh",
                              choices=["fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky", "bonferroni", "holm"]) ## include the fwer (family-wise error rate) options - bonferroni, sidak, holm, holm-sidak, simes-hochberg, hommel?
    between_corr.add_argument("-s", "--min_sample", 
                              help="minimum number of samples present in", 
                              type=int)
    between_corr.add_argument("--max_p", 
                              help="max p-value to determine edges", 
                              type=float)
    between_corr.add_argument("--min_r", 
                              help="minimum R to determine edges", 
                              type=float)
    between_corr.add_argument("--sparcc_filter", 
                              help="filter using parameters from SparCC publication", 
                              default=False,
                              action="store_true")
    between_corr.add_argument("--procs", 
                              help="number of processors to use", 
                              default=1, 
                              type=int)
    between_corr.add_argument("-f", "--force", 
                              help="force overwrite output folder if it already exists", 
                              default=False,
                              action="store_true")
    between_corr.add_argument("-v", "--verbose", 
                             help="give verbose messages to STDOUT", 
                             default=False, 
                             action="store_true")
    between_corr.set_defaults(func=between_correls)

    return parser


def main(argv=None):
    """Parse CLI arguments and run the selected SCNIC workflow"""
    parser = build_parser()
    args = parser.parse_args(argv)

    args_dict = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "func"}
    }
    args.func(**args_dict)


if __name__ == "__main__":
    main()
