__author__ = 'shafferm'

import argparse
from biom import load_table
from correl_nets.module_maker import sparcc_correlations, sparcc_correlations_multi
from correl_nets.general import print_delimited

if __name__ == '__main__':
    """main, takes argparser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="location of input biom file")
    parser.add_argument("-o", "--output", help="output file location", default="correls.txt")
    parser.add_argument("-s", "--single", help="multiprocess", action='store_true', default=False)
    parser.add_argument("-b", "--boots", help="number of bootstraps", type=int, default=100)
    args = parser.parse_args()

    table = load_table(args.input)
    if args.single:
        correls, header = sparcc_correlations(table, bootstraps=args.boots)
    else:
        correls, header = sparcc_correlations_multi(table, bootstraps=args.boots)
    print_delimited(args.out, correls, header)
