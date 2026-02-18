#!/usr/bin/env python
"""
Generate a table with combined model inference benchmarks.
"""

import pandas as pd


# Path to the result directory. Adjust as needed.
resdir = 'results'

# Methods being compared.
methods = ['Lobster 24M', 'Lobster 150M', 'ProtT5-XL', 'Ankh Base', 'Ankh Large', 'ProtMamba']


def main():
    dfs = []
    for method in methods:
        df = pd.read_csv(f'{resdir}/{method}_results.csv', index_col=0).rename(
            columns={'Ms per protein': method}).set_index(
                ['Sequence Length', 'Batch Size'])
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    print(df.shape[0])
    df.to_csv('results.tsv', sep='\t')


if __name__ == '__main__':
    main()
