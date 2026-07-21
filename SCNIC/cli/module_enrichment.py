#!/usr/local/bin/python3

"""
Module enrichment command-line wrapper (called using module-enrichment).

This script exposes three subcommands for module-enrichment workflows:
- annotate   : annotate a correlation file with tree/genome/module metadata
- perms      : generate permutation statistics for module membership
- stats      : calculate summary statistics and generate plots

Each subcommand accepts its own CLI options documented below.
"""

import os
os.environ['OMP_NUM_THREADS'] = '8' ## should this be changed?

import argparse
from SCNIC.annotate_correls import do_annotate_correls
from SCNIC.calculate_permutations import do_multiprocessed_perms
from SCNIC.calculate_permutation_stats import do_stats

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


def main():
    """
    Parse command-line arguments and execute module-enrichment workflows.

    Subcommands
    -----------
    annotate
        --correls : correlation file to annotate
        --tree    : tree file path
        --genome  : genome metadata file path
        --modules : module definitions path
        --output  : annotation output path
        --skip_kos: skip PD KO computation
        --to_keep : optional module filter file

    perms
        --correls : correlation table path
        --perms   : number of permutations
        --procs   : number of processes
        --modules : module definitions path
        --output  : output directory
        --skip_kos: skip KO output
        --to_keep : optional module filter file

    stats
        --correls : correlation table path
        --modules : module definitions path
        --perms   : permutation results directory
        --output  : output directory
        --skip_kos: skip KO stats
        --to_keep : optional module filter file
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        action="version",
        version=f"{__version__}",
    )

    subparsers = parser.add_subparsers(dest='subparser_name')
    annotate_correls = subparsers.add_parser("annotate", 
                                             help="annotate correls.txt file",
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_perms = subparsers.add_parser("perms", 
                                      help="run permutation generation",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    calc_stats = subparsers.add_parser("stats", 
                                       help="generate p-values from the permutations",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # annotate correlations
    annotate_correls.add_argument('--correls')
    annotate_correls.add_argument('--tree')
    annotate_correls.add_argument('--genome')
    annotate_correls.add_argument('--modules')
    annotate_correls.add_argument('--output')
    annotate_correls.add_argument('--skip_kos', 
                                  default=False, 
                                  action='store_true')
    annotate_correls.add_argument('--to_keep')

    # run permutations multiprocessed
    run_perms.add_argument('--correls')
    run_perms.add_argument('--perms', 
                           type=int, 
                           default=2000)
    run_perms.add_argument('--procs', 
                           type=int, 
                           default=4)
    run_perms.add_argument('--modules')
    run_perms.add_argument('--output')
    run_perms.add_argument('--skip_kos', 
                           default=False, 
                           action='store_true')
    run_perms.add_argument('--to_keep')

    # calculate statistics
    calc_stats.add_argument('--correls')
    calc_stats.add_argument('--modules')
    calc_stats.add_argument('--perms')
    calc_stats.add_argument('--output')
    calc_stats.add_argument('--skip_kos', 
                            default=False, 
                            action='store_true')
    calc_stats.add_argument('--to_keep')

    args = parser.parse_args()

    if args.subparser_name == 'annotate':
        do_annotate_correls(args.correls, 
                            args.tree, 
                            args.genome, 
                            args.modules, 
                            args.output, 
                            args.skip_kos,
                            args.to_keep)
    elif args.subparser_name == 'perms':
        do_multiprocessed_perms(args.correls, 
                                args.perms, 
                                args.procs, 
                                args.modules, 
                                args.output, 
                                args.skip_kos,
                                args.to_keep)
    elif args.subparser_name == 'stats':
        do_stats(args.correls, 
                 args.modules, 
                 args.perms, 
                 args.output, 
                 args.skip_kos, 
                 args.to_keep)
    else:
        print('What the hell happened here?')


if __name__ == '__main__':
    main()
