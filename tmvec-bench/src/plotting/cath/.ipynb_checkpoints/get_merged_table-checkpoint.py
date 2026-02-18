#!/usr/bin/env python
"""
Generate a table with combined CATH S100 analysis results.
"""

import pandas as pd


# Path to the result directory. Adjust as needed.
resdir = 'results'

# Methods being compared.
methods = ['tmvec1', 'tmvec2', 'tmvec2_student', 'tmalign', 'foldseek']


def main():
    dfs = []
    for method in methods[:4]:
        dfs.append(pd.read_csv(f'{resdir}/cath_{method}_similarities.csv').rename(
            columns={'tm_score': method}))

    # Foldseek reports both TM-scores and E-values. We will use E-values.
    dfs.append(pd.read_csv(f'{resdir}/cath_foldseek_similarities.csv').drop(
        columns='tm_score').rename(columns={'evalue': 'foldseek'}))

    # TM-Vec models generate a suffix like "/1-150" after sequence ID. Remove it.
    for df in dfs[:3]:
        for i in (1, 2):
            df[f'seq{i}_id'] = df[f'seq{i}_id'].str.split('/').str[0]

    # Remove sequence ID prefix like "cath|4_4_0|".
    for df in dfs:
        for i in (1, 2):
            df[f'seq{i}_id'] = df[f'seq{i}_id'].str.split('|').str[2]

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
