import argparse

from biom import load_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="location of input biom file")
    parser.add_argument("-o", "--output", help="output biom file")
    parser.add_argument("-n", "--min_percent", help="minimum decimal percentatge present", type=float)
    parser.add_argument("-s", "--min_samples", help="minimum number of samples present")
    args = parser.parse_args()

    table = load_table(args.input)

    # first sample filter
    if args.min_samples != None:
        table.filter([i for i in table.ids(axis='observation') if sum(table.data(i, axis='observation') != 0) > 10], axis='observation')

    if args.min_percent != None:
        # set values below min_percent to zero
        for i in xrange(table.shape[0]):
            for j in xrange(table.shape[1]):
                if table.matrix_data[i,j] < args.min_percent:
                    table.matrix_data[i,j] = 0

        # filter out rows of all zeroes
        table.filter([i for i in table.ids(axis='observation') if sum(table.data(i, axis='observation'))!=0], axis='observation')

        # second sample filter
        if args.min_samples != None:
            table.filter([i for i in table.ids(axis='observation') if sum(table.data(i, axis='observation') != 0) > 10], axis='observation')

    table.to_json('filtered relative abundance', open(args.output,'w'))

if __name__ == '__main__':
    main()
