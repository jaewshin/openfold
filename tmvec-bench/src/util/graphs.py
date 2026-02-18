#!/usr/bin/env python
"""Density scatter plots for benchmarking method comparisons."""

import argparse
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme()


def plot_density_scatter(df, pred_col, truth_col, method_name, dataset_name, output_path=None):
    """Create density scatter plot comparing predicted vs ground truth."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Hexbin density plot with rocket colormap
    hb = ax.hexbin(
        df[truth_col],
        df[pred_col],
        gridsize=100,
        cmap='rocket',
        mincnt=1
    )

    # Add identity line
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=2)

    # Generate title with dataset name
    title = f'{dataset_name} Alignment Results ({method_name})'

    ax.set_title(title, fontsize=20, fontweight='bold', pad=20, fontfamily='sans-serif')
    ax.set_xlabel('TM-align Ground Truth', fontsize=16, fontweight='bold')
    ax.set_ylabel(method_name, fontsize=16, fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Calculate Pearson R
    pearson_r, _ = stats.pearsonr(df[truth_col], df[pred_col])

    # Add total count and Pearson R box
    total_count = len(df)
    stats_text = f'n = {total_count:,}\nPearson R = {pearson_r:.3f}'
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Add colorbar
    plt.colorbar(hb, ax=ax, label='Count')
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')

    return fig


def main():
    parser = argparse.ArgumentParser(description='Generate density scatter plots for method comparison')
    parser.add_argument('method', choices=['foldseek', 'tmvec1', 'tmvec2', 'tmvec2_student'],
                        help='Method to compare against TM-align')
    parser.add_argument('--tmalign', help='Path to TM-align results (auto-detected if not provided)')
    parser.add_argument('--method-file', help='Path to method results (auto-detected if not provided)')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Maximum number of pairs to load')
    parser.add_argument('--output-dir', help='Output directory (default: figures/{method})')
    args = parser.parse_args()

    # Method display names
    method_config = {
        'foldseek': 'Foldseek',
        'tmvec1': 'TMvec-1',
        'tmvec2': 'TMvec-2',
        'tmvec2_student': 'TMvec-2 Student'
    }

    # Auto-detect or construct method file path
    if args.method_file:
        method_file = args.method_file
    else:
        # Default to cath dataset
        method_file = f'results/cath_{args.method}_similarities.csv'

    # Detect dataset from method file path
    is_scope40 = 'scope40' in Path(method_file).name
    dataset_name = 'SCOPe40' if is_scope40 else 'CATH'
    dataset_prefix = 'scope40_' if is_scope40 else 'cath_'

    # Auto-detect tmalign file based on dataset
    if args.tmalign:
        tmalign_file = args.tmalign
    else:
        tmalign_file = f'results/{dataset_prefix}tmalign_similarities.csv'

    # Load data - support both CSV and parquet
    def load_file(filepath):
        if filepath.endswith('.parquet'):
            return pq.read_table(filepath).to_pandas()
        else:
            return pd.read_csv(filepath)

    df_tmalign = load_file(tmalign_file)
    if 'tm_score' in df_tmalign.columns:
        df_tmalign['tm_score'] = pd.to_numeric(df_tmalign['tm_score'], errors='coerce')

    df_method = load_file(method_file)
    if args.max_pairs is not None:
        df_method = df_method.head(args.max_pairs)
    if 'tm_score' in df_method.columns:
        df_method['tm_score'] = pd.to_numeric(df_method['tm_score'], errors='coerce')

    # Clean sequence IDs
    # Remove cath|CLASS| prefix if present (e.g., cath|4_4_0|107lA00 -> 107lA00)
    # Remove /RANGE suffix if present (e.g., 107lA00/1-162 -> 107lA00)
    df_method['seq1_clean'] = df_method['seq1_id'].str.replace(r'cath\|[^|]+\|', '', regex=True).str.replace(r'/\d+-\d+', '', regex=True)
    df_method['seq2_clean'] = df_method['seq2_id'].str.replace(r'cath\|[^|]+\|', '', regex=True).str.replace(r'/\d+-\d+', '', regex=True)

    # Merge
    df_merged = pd.merge(
        df_tmalign[['seq1_id', 'seq2_id', 'tm_score']],
        df_method[['seq1_clean', 'seq2_clean', 'tm_score']],
        left_on=['seq1_id', 'seq2_id'],
        right_on=['seq1_clean', 'seq2_clean'],
        suffixes=('_tmalign', '_method')
    )

    print(f"Plotting {len(df_merged)} pairs")

    # Output - include dataset in path to avoid overwriting
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dataset_subdir = dataset_prefix.rstrip('_') if dataset_prefix else 'cath'
        output_dir = Path(f'figures/{dataset_subdir}/{args.method}')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_density_scatter(
        df_merged,
        'tm_score_method',
        'tm_score_tmalign',
        method_config[args.method],
        dataset_name,
        output_dir / 'density_scatter'
    )
    print(f"Saved to {output_dir}/density_scatter.png")


if __name__ == "__main__":
    main()
