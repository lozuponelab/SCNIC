#!/usr/bin/env python

__author__ = 'shafferm'

"""
merge_biom_tables.py

Takes two biom tables and merges them.  Assumes that samples are all the same and observations are all different.
"""

import argparse
import sys

from biom import load_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-1", "--table1", help="First biom table")
    parser.add_argument("-2", "--table2", help="Second biom table")
    parser.add_argument("-o", "--output", help="Output biom table")
    args = parser.parse_args()

    table1 = load_table(args.table1)
    table2 = load_table(args.table2)

    # check that samples are the same
    if set(table1.ids()) != set(table2.ids()):
        print "biom tables have different sample names"
        sys.exit()

    merged_table = table1.merge(table2)

    # check that dimensions of new table are correct
    if table1.shape[1] != merged_table.shape[1]:
        print "merged_table has different number of samples"
        sys.exit()
    if table1.shape[0]+table2.shape[0] != merged_table.shape[0]:
        print "merged_table has different number of observations"
        sys.exit()

    with open(args.output, 'w') as f:
        f.write(merged_table.to_json("merge_tables.py"))

if __name__ == '__main__':
    main()
