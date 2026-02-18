#!/usr/bin/env python
"""
Generate a table with combined SCOPe 40 analysis results.
"""

import pandas as pd


# Path to the result directory. Adjust as needed.
resdir = 'results'

# Methods being compared.
methods = ['tmvec1', 'tmvec2', 'tmvec2_student', 'tmalign', 'foldseek']


def main():
    dfs = []
    for method in methods[:4]:
        df = pd.read_csv(f'{resdir}/scope40_{method}_similarities.csv').rename(
            columns={'tm_score': method})
        dfs.append(df)

    # Foldseek reports both TM-scores and E-values. We will use E-values.
    df = pd.read_csv(f'{resdir}/scope40_foldseek_similarities.csv').drop(
        columns='tm_score').rename(columns={'evalue': 'foldseek'})

    # Two domains were broken down into chains in the Foldseek analysis, and
    # they cannot be compared with other results. Remove them.
    with open('domain.lst', 'r') as fh:
        domains = fh.read().splitlines()
    df.query('seq1_id in @domains & seq2_id in @domains', inplace=True)
    dfs.append(df)

    # Sort and merge IDs of sequences 1 and 2.
    for df in dfs:
        df['seq_pair'] = df.apply(lambda row: ','.join(sorted(
            [row['seq1_id'], row['seq2_id']])), axis=1)
        df.drop(['seq1_id', 'seq2_id'], axis=1, inplace=True)
        df.set_index('seq_pair', inplace=True)

    # Combine results.
    df = pd.concat(dfs, axis=1)
    print(df.shape[0])
    # df.dropna(how='any', inplace=True)

    # Append ground truth.
    truth = pd.read_table('truth.tsv')
    truth['seq_pair'] = truth['a'] + ',' + truth['b']
    truth.set_index('seq_pair', inplace=True)
    truth.drop(columns=['a', 'b'], inplace=True)
    df = pd.concat([truth, df], axis=1)

    # Output.
    df.to_csv('results.tsv', sep='\t')


if __name__ == '__main__':
    main()
