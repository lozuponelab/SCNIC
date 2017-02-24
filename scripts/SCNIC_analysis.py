#!/usr/local/bin/python2

import argparse
from SCNIC.within_correls import within_correls
from SCNIC.between_correls import between_correls

__author__ = 'shafferm'

"""Entry to both module_maker and between_correls, only holds main and args are passed to the corresponding program
"""


def main():
    """Things"""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    within_corr = subparsers.add_parser("within", help="Find pairwise correlations within a table and make modules",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    between_corr = subparsers.add_parser("between", help="Find correlations between two tables",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    within_corr.add_argument("-i", "--input", help="location of input biom file", required=True)
    within_corr.add_argument("-o", "--output", help="output file location")
    within_corr.add_argument("-m", "--correl_method", help="correlation method", default="sparcc")
    within_corr.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    within_corr.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    within_corr.add_argument("--prefix", help="prefix for module names in collapsed file", default="module_")
    within_corr.add_argument("--k_size", help="desired k-size to determine cliques", default=3, type=int)
    within_corr.add_argument("--min_p", help="minimum p-value to determine edges", type=float)
    within_corr.add_argument("--min_r", help="minimum correlation value to determine edges", type=float)
    within_corr.add_argument("--outlier_removal", help="outlier detection and removal", default=False,
                             action="store_true")
    within_corr.add_argument("--procs", help="number of processors to use", default=1, type=int)
    within_corr.add_argument("-b", "--bootstraps", help="number of bootstraps", default=100, type=int)
    within_corr.add_argument("-e", "--rarefaction_level", help="level of rarefaction", default=1000, type=int)
    within_corr.add_argument("-f", "--force", help="force overwrite output folder if it already exists", default=False,
                             action="store_true")
    within_corr.add_argument("--sparcc_filter", help="filter as described in SparCC paper", default=False,
                             action="store_true")
    within_corr.add_argument("--verbose", help="give verbose messages to STDOUT", default=False, action="store_true")
    within_corr.set_defaults(func=within_correls)

    between_corr.add_argument("-1", "--table1", help="table to be correlated", required=True)
    between_corr.add_argument("-2", "--table2", help="second table to be correlated", required=True)
    between_corr.add_argument("-o", "--output", help="output file location")
    between_corr.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    between_corr.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    between_corr.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    between_corr.add_argument("--min_p", help="minimum p-value to determine edges", type=float)
    between_corr.add_argument("--min_r", help="minimum R to determine edges", type=float)
    between_corr.set_defaults(func=between_correls)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
