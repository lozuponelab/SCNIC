import argparse
from biom import load_table

__author__ = 'shafferm'

"""script that takes an input biom file of otus and strips it to only contain closed reference picked otus"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    table = load_table(args.input)

    ids_to_keep = list()
    for id in table.ids('observation'):
        if not id.startswith('New'):
            ids_to_keep.append(id)
    table.filter(ids_to_keep, axis="observation")

    with open(args.output, 'w') as f:
        f.write(table.to_json("filt_to_closed.py"))

if __name__ == '__main__':
    main()
