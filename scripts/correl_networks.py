#!/usr/local/bin/python2

__author__ = 'shafferm'

"""Entry to both module_maker and between_correls, only holds main and args are passed to the corresponding program
"""

import argparse
from correl_nets.module_maker import module_maker
from correl_nets.between_correls import between_correls


def main():
    """Things"""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    within_corr = subparsers.add_parser("within", help="Find pairwise correlations within a table and make modules")
    between_corr = subparsers.add_parser("between", help="Find correlations between two tables")

    within_corr.add_argument("-i", "--input", help="location of input biom file")
    within_corr.add_argument("-o", "--output", help="output file location")
    within_corr.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    within_corr.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    within_corr.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    within_corr.add_argument("--prefix", help="prefix for module names in collapsed file", default="module_")
    within_corr.add_argument("-k", "--k_size", help="desired k-size to determine cliques", default=3, type=int)
    within_corr.add_argument("--min_p", help="minimum p-value to determine edges", default=.05, type=float)
    within_corr.add_argument("--outlier_removal", help="outlier detection and removal", default=False, action="store_false")
    within_corr.add_argument("--procs", help="number of processors for sparcc", default=None)
    within_corr.add_argument("-b", "--bootstraps", help="number of bootstraps", default=100, type=int)
    within_corr.set_defaults(func=module_maker)

    between_corr.add_argument("-1", "--table1", help="table to be correlated", required=True)
    between_corr.add_argument("-2", "--table2", help="second table to be correlated", required=True)
    between_corr.add_argument("-o", "--output", help="output file location")
    between_corr.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    between_corr.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    between_corr.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    between_corr.add_argument("--min_p", help="minimum p-value to determine edges", default=.05, type=float)
    between_corr.set_defaults(func=between_correls)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
