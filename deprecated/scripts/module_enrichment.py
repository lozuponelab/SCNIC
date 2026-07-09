#!/usr/local/bin/python3

import os
os.environ['OMP_NUM_THREADS'] = '8'

import argparse
from SCNIC.annotate_correls import do_annotate_correls
from SCNIC.calculate_permutations import do_multiprocessed_perms
from SCNIC.calculate_permutation_stats import do_stats


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='subparser_name')
    annotate_correls = subparsers.add_parser("annotate", help="Annotate correls.txt file",
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_perms = subparsers.add_parser("perms", help="Run permutation generation",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    calc_stats = subparsers.add_parser("stats", help="Generate p-values from the permutations",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # annotate correlations
    annotate_correls.add_argument('--correls')
    annotate_correls.add_argument('--tree')
    annotate_correls.add_argument('--genome')
    annotate_correls.add_argument('--modules')
    annotate_correls.add_argument('--output')
    annotate_correls.add_argument('--skip_kos', default=False, action='store_true')
    annotate_correls.add_argument('--to_keep')

    # run permutations multiprocessed
    run_perms.add_argument('--correls')
    run_perms.add_argument('--perms', type=int, default=2000)
    run_perms.add_argument('--procs', type=int, default=4)
    run_perms.add_argument('--modules')
    run_perms.add_argument('--output')
    run_perms.add_argument('--skip_kos', default=False, action='store_true')
    run_perms.add_argument('--to_keep')

    # calculate statistics
    calc_stats.add_argument('--correls')
    calc_stats.add_argument('--modules')
    calc_stats.add_argument('--perms')
    calc_stats.add_argument('--output')
    calc_stats.add_argument('--skip_kos', default=False, action='store_true')
    calc_stats.add_argument('--to_keep')

    args = parser.parse_args()

    if args.subparser_name == 'annotate':
        do_annotate_correls(args.correls, args.tree, args.genome, args.modules, args.output, args.skip_kos,
                            args.to_keep)
    elif args.subparser_name == 'perms':
        do_multiprocessed_perms(args.correls, args.perms, args.procs, args.modules, args.output, args.skip_kos,
                                args.to_keep)
    elif args.subparser_name == 'stats':
        do_stats(args.correls, args.modules, args.perms, args.output, args.skip_kos, args.to_keep)
    else:
        print('What the hell happened here?')


if __name__ == '__main__':
    main()
