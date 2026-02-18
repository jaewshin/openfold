#!/usr/bin/env python
"""
Generate ground truth matches between SCOPe 40 proteins.

Notes:
    SCOPe sequence data (clustered at 40% sequence identity) were retrieved from:
    https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/

    Which is the data repository supporting the Foldseek paper:
    https://www.nature.com/articles/s41587-023-01773-0

    SCOPe 2.01 classifications were retrieved from the official SCOPe website:
    https://scop.berkeley.edu/downloads/parse/dir.des.scope.2.01-stable.txt

"""

from itertools import combinations
import pandas as pd


def main():
    levels = ['class', 'fold', 'superfamily', 'family']
    columns = ['sunid', 'sid', 'family', 'domain', 'description']
    df = pd.read_table('dir.des.scope.2.01-stable.txt', names=columns, comment='#')
    df = df.query('domain != "-"').set_index('domain')

    assert (df['family'].str.split('.').str.len() == 4).all()
    df['superfamily'] = df['family'].str.rsplit('.', n=1).str[0]
    df['fold'] = df['superfamily'].str.rsplit('.', n=1).str[0]
    df['class'] = df['fold'].str.rsplit('.', n=1).str[0]

    with open('domain.lst', 'r') as fh:
        domains = fh.read().splitlines()
    df = df.loc[domains]

    with open('truth.tsv', 'w') as fh:
        print('a', 'b', *levels, sep='\t', file=fh)
        for a, b in combinations(domains, 2):
            out = [a, b]
            for level in levels:
                out.append(str(int(df.loc[a, level] == df.loc[b, level])))
            print(*out, sep='\t', file=fh)


if __name__ == '__main__':
    main()
