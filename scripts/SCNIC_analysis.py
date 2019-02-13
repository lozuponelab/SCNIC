#!/usr/local/bin/python3

import argparse
from SCNIC.within_correls import within_correls
from SCNIC.between_correls import between_correls
from SCNIC.module import module_maker

__author__ = 'shafferm'

"""Entry to both module_maker and between_correls, only holds main and args are passed to the corresponding program"""


def main():
    """Things"""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    within_corr = subparsers.add_parser("within", help="Find pairwise correlations within a table",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    make_modules = subparsers.add_parser("modules", help="Make modules on a network",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    between_corr = subparsers.add_parser("between", help="Find correlations between two tables",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser for making correlation network with a single biom table
    within_corr.add_argument("-i", "--input_loc", help="location of input biom file", required=True)
    within_corr.add_argument("-o", "--output_loc", help="output directory")
    within_corr.add_argument("-m", "--correl_method", help="correlation method", default="sparcc")
    within_corr.add_argument("-a", "--p_adjust", help="p-value adjustment", default="fdr_bh")
    within_corr.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    within_corr.add_argument("--procs", help="number of processors to use", default=1, type=int)
    within_corr.add_argument("--sparcc_filter", help="filter as described in SparCC paper", default=False,
                             action="store_true")
    within_corr.add_argument("--sparcc_p", help="Calculate p-value for sparCC R value, give number of bootstraps",
                             type=int)
    within_corr.add_argument("--verbose", help="give verbose messages to STDOUT", default=False, action="store_true")
    within_corr.set_defaults(func=within_correls)

    # parser for finding modules in a correlation network
    make_modules.add_argument("-i", "--input_loc", help="location of output from SCNIC_analysis.py within",
                              required=True)
    make_modules.add_argument("-o", "--output_loc", help="output directory")
    make_modules.add_argument("--min_p", help="minimum p-value to determine edges, p must have been calculated",
                              type=float)
    make_modules.add_argument("--min_r", help="minimum correlation value to determine edges", type=float)
    make_modules.add_argument("--method", help="method to be used for determining modules, pick from: naive, k_cliques "
                                               "or louvain", default='naive')
    make_modules.add_argument("-k", "--k_size", help="k value for use with the k-clique communities algorithm",
                              type=int, default=3)
    make_modules.add_argument("-g", "--gamma", help="gamma value for use with louvain modularity maximization, between "
                                                    "0 and 1 where 0 makes small modules and 1 large modules",
                              type=float, default=0.1)
    make_modules.add_argument("--table_loc", help="biom table used to make network to be collapsed")
    make_modules.add_argument("--prefix", help="prefix for module names in collapsed file", default="module")
    make_modules.add_argument("-v", "--verbose", help="give verbose messages to STDOUT", default=False,
                              action="store_true")
    make_modules.set_defaults(func=module_maker)

    # parser for building a bipartite correlation network between two data types
    between_corr.add_argument("-1", "--table1", help="table to be correlated", required=True)
    between_corr.add_argument("-2", "--table2", help="second table to be correlated", required=True)
    between_corr.add_argument("-o", "--output_loc", help="output file location")
    between_corr.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    between_corr.add_argument("-a", "--p_adjust", help="p-value adjustment", default="fdr_bh")
    between_corr.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    between_corr.add_argument("--max_p", help="minimum p-value to determine edges", type=float)
    between_corr.add_argument("--min_r", help="minimum R to determine edges", type=float)
    between_corr.add_argument("--sparcc_filter", help="filter using parameters from SparCC publication", default=False,
                              action="store_true")
    between_corr.add_argument("--procs", help="number of processors to use", default=1, type=int)
    between_corr.add_argument("-f", "--force", help="force overwrite output folder if it already exists", default=False,
                              action="store_true")
    between_corr.set_defaults(func=between_correls)

    args = parser.parse_args()
    args_dict = {i: j for i, j in vars(args).items() if i != 'func'}
    args.func(**args_dict)


if __name__ == "__main__":
    main()
