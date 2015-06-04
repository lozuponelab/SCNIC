#!/usr/local/bin/python2

__author__ = 'shafferm'

"""Entry to both module_maker and between_correls, only holds main and args are passed to the corresponding program
"""

import argparse
import module_maker
import between_correls


def main():
    """Things"""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    within_corr = subparsers.add_parser()
    between_corr = subparsers.add_parser()

    within_corr.add_argument("-i", "--input", help="location of input biom file")
    within_corr.add_argument("-o", "--output", help="output file location")
    within_corr.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    within_corr.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    within_corr.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    within_corr.add_argument("--prefix", help="prefix for module names in collapsed file", default="module_")
    within_corr.add_argument("-k", "--k_size", help="desired k-size to determine cliques", default=3, type=int)
    within_corr.set_defaults(func=module_maker.main)

    between_corr.add_argument("-1", "--table1", help="table to be correlated")
    between_corr.add_argument("-2", "--table2", help="second table to be correlated")
    between_corr.add_argument("-o", "--output", help="output file location")
    between_corr.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    between_corr.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    between_corr.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    between_corr.set_defaults(func=between_correls.main)

    args = parser.parse_args()
    args.func(args)
