#!/usr/bin/env python
"""Density scatter plots for benchmarking method comparisons."""

import argparse
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme()


def load_dataframe(filepath):
    """Load dataframe from either CSV or Parquet file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix.lower() == '.csv':
        print(f"Loading CSV: {filepath}")
        return pd.read_csv(filepath)
    elif filepath.suffix.lower() in ['.parquet', '.pq']:
        print(f"Loading Parquet: {filepath}")
        return pq.read_table(filepath).to_pandas()
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .csv or .parquet")


def compute_detailed_statistics(df, pred_col, truth_col, method_name):
    """Compute comprehensive statistics for method evaluation."""
    pred = df[pred_col].values
    truth = df[truth_col].values
    
    # Basic correlation metrics
    pearson_r, pearson_p = stats.pearsonr(truth, pred)
    spearman_r, spearman_p = stats.spearmanr(truth, pred)
    
    # Error metrics
    mae = mean_absolute_error(truth, pred)
    mse = mean_squared_error(truth, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(truth, pred)
    
    # Residual analysis
    residuals = pred - truth
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Classification accuracy at 0.5 threshold (fold similarity)
    truth_high = truth >= 0.5
    pred_high = pred >= 0.5
    
    # Confusion matrix components
    true_positives = np.sum((truth_high) & (pred_high))
    true_negatives = np.sum((~truth_high) & (~pred_high))
    false_positives = np.sum((~truth_high) & (pred_high))
    false_negatives = np.sum((truth_high) & (~pred_high))
    
    total = len(truth)
    accuracy = (true_positives + true_negatives) / total
    
    # Precision, Recall, F1 (for TM >= 0.5)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Distribution statistics
    truth_stats = {
        'mean': np.mean(truth),
        'median': np.median(truth),
        'std': np.std(truth),
        'min': np.min(truth),
        'max': np.max(truth),
        'q25': np.percentile(truth, 25),
        'q75': np.percentile(truth, 75)
    }
    
    pred_stats = {
        'mean': np.mean(pred),
        'median': np.median(pred),
        'std': np.std(pred),
        'min': np.min(pred),
        'max': np.max(pred),
        'q25': np.percentile(pred, 25),
        'q75': np.percentile(pred, 75)
    }
    
    # Stratified accuracy by TM-score ranges
    ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    range_stats = {}
    for low, high in ranges:
        mask = (truth >= low) & (truth < high)
        if np.sum(mask) > 0:
            range_mae = mean_absolute_error(truth[mask], pred[mask])
            range_pearson = stats.pearsonr(truth[mask], pred[mask])[0]
            range_stats[f'{low}-{high}'] = {
                'count': np.sum(mask),
                'mae': range_mae,
                'pearson': range_pearson
            }
    
    return {
        'method': method_name,
        'n_pairs': total,
        'correlation': {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'r2': r2
        },
        'error_metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mean_residual': mean_residual,
            'std_residual': std_residual
        },
        'classification_0.5': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        },
        'ground_truth_distribution': truth_stats,
        'prediction_distribution': pred_stats,
        'stratified_performance': range_stats
    }


def print_statistics(stats_dict, verbose=True):
    """Pretty print statistics dictionary."""
    print("\n" + "="*80)
    print(f"PERFORMANCE STATISTICS: {stats_dict['method']}")
    print("="*80)
    
    print(f"\nDataset Size: {stats_dict['n_pairs']:,} protein pairs")
    
    # Correlation metrics
    print("\n--- CORRELATION METRICS ---")
    corr = stats_dict['correlation']
    print(f"  Pearson R:     {corr['pearson_r']:.4f}  (p={corr['pearson_p']:.2e})")
    print(f"  Spearman R:    {corr['spearman_r']:.4f}  (p={corr['spearman_p']:.2e})")
    print(f"  R² Score:      {corr['r2']:.4f}")
    
    # Error metrics
    print("\n--- ERROR METRICS ---")
    err = stats_dict['error_metrics']
    print(f"  MAE:           {err['mae']:.4f}")
    print(f"  RMSE:          {err['rmse']:.4f}")
    print(f"  Mean Residual: {err['mean_residual']:+.4f}")
    print(f"  Std Residual:  {err['std_residual']:.4f}")
    
    # Classification at 0.5 threshold
    print("\n--- CLASSIFICATION PERFORMANCE (TM ≥ 0.5) ---")
    clf = stats_dict['classification_0.5']
    print(f"  Accuracy:      {clf['accuracy']:.4f}")
    print(f"  Precision:     {clf['precision']:.4f}")
    print(f"  Recall:        {clf['recall']:.4f}")
    print(f"  F1 Score:      {clf['f1']:.4f}")
    
    if verbose:
        print("\n  Confusion Matrix:")
        print(f"    True Positives:  {clf['true_positives']:,}")
        print(f"    True Negatives:  {clf['true_negatives']:,}")
        print(f"    False Positives: {clf['false_positives']:,}")
        print(f"    False Negatives: {clf['false_negatives']:,}")
        
        # Distribution statistics
        print("\n--- DISTRIBUTION STATISTICS ---")
        print("\n  Ground Truth (TM-align):")
        truth_dist = stats_dict['ground_truth_distribution']
        print(f"    Mean:   {truth_dist['mean']:.4f} ± {truth_dist['std']:.4f}")
        print(f"    Median: {truth_dist['median']:.4f}")
        print(f"    Range:  [{truth_dist['min']:.4f}, {truth_dist['max']:.4f}]")
        print(f"    IQR:    [{truth_dist['q25']:.4f}, {truth_dist['q75']:.4f}]")
        
        print("\n  Predictions:")
        pred_dist = stats_dict['prediction_distribution']
        print(f"    Mean:   {pred_dist['mean']:.4f} ± {pred_dist['std']:.4f}")
        print(f"    Median: {pred_dist['median']:.4f}")
        print(f"    Range:  [{pred_dist['min']:.4f}, {pred_dist['max']:.4f}]")
        print(f"    IQR:    [{pred_dist['q25']:.4f}, {pred_dist['q75']:.4f}]")
        
        # Stratified performance
        print("\n--- STRATIFIED PERFORMANCE BY TM-SCORE RANGE ---")
        print(f"  {'Range':<12} {'Count':<10} {'MAE':<10} {'Pearson R':<10}")
        print("  " + "-"*42)
        for range_name, range_stats in stats_dict['stratified_performance'].items():
            print(f"  {range_name:<12} {range_stats['count']:<10,} "
                  f"{range_stats['mae']:<10.4f} {range_stats['pearson']:<10.4f}")
    
    print("\n" + "="*80 + "\n")


def plot_density_scatter(df, pred_col, truth_col, method_name, output_path=None):
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

    # Determine title based on method
    if 'foldseek' in method_name.lower():
        title = 'SCOPe40 Alignment Results (Foldseek)'
    elif 'tmvec-1' in method_name.lower():
        title = 'SCOPe40 Alignment Results (TMvec-1)'
    elif 'student' in method_name.lower():
        title = 'SCOPe40 Alignment Results (TMvec-Student)'
    else:
        title = f'Alignment Results ({method_name})'

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
    parser.add_argument('method', choices=['foldseek', 'tmvec1', 'tmvec2', 'student'],
                        help='Method to compare against TM-align')
    parser.add_argument('--tmalign', default='results/tmalign_similarities.parquet',
                        help='Path to TM-align results')
    parser.add_argument('--method-file', help='Path to method results (auto-detected if not provided)')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Maximum number of pairs to load')
    parser.add_argument('--output-dir', help='Output directory (default: figures/{method})')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed performance statistics')
    parser.add_argument('--save-stats', help='Save statistics to JSON file')
    args = parser.parse_args()

    # Auto-detect method file and display name (supports both .csv and .parquet)
    method_config = {
        'foldseek': {
            'file': 'results/scope40_foldseek_similarities',
            'name': 'Foldseek',
            'tmalign_default': 'results/scope40_tmalign_similarities'
        },
        'tmvec1': {
            'file': 'results/scope40_tmvec1_similarities',
            'name': 'TMvec-1',
            'tmalign_default': 'results/scope40_tmalign_similarities'
        },
        'tmvec2': {
            'file': 'results/tmvec2_similarities',
            'name': 'TMvec-2',
            'tmalign_default': 'results/tmalign_similarities'
        },
        'student': {
            'file': 'results/tmvec_student_similarities',
            'name': 'TMvec-Student',
            'tmalign_default': 'results/tmalign_similarities'
        }
    }
    
    # Auto-detect file extension (.csv or .parquet)
    def find_file_with_extension(base_path):
        """Try to find file with .csv or .parquet extension."""
        for ext in ['.csv', '.parquet', '.pq']:
            path = Path(f"{base_path}{ext}")
            if path.exists():
                return str(path)
        # Return original with .csv as fallback
        return f"{base_path}.csv"

    config = method_config[args.method]
    
    # Auto-detect file extensions if not explicitly provided
    if args.method_file:
        method_file = args.method_file
    else:
        method_file = find_file_with_extension(config['file'])
    
    if args.tmalign != 'results/tmalign_similarities.parquet':
        tmalign_file = args.tmalign
    else:
        tmalign_file = find_file_with_extension(config['tmalign_default'])

    # Load data (supports both CSV and Parquet)
    df_tmalign = load_dataframe(tmalign_file)
    if 'tm_score' in df_tmalign.columns:
        df_tmalign['tm_score'] = pd.to_numeric(df_tmalign['tm_score'], errors='coerce')

    df_method = load_dataframe(method_file)
    if args.max_pairs is not None:
        df_method = df_method.head(args.max_pairs)
    if 'tm_score' in df_method.columns:
        df_method['tm_score'] = pd.to_numeric(df_method['tm_score'], errors='coerce')

    # Clean sequence IDs
    df_method['seq1_clean'] = df_method['seq1_id'].str.replace(r'/\d+-\d+', '', regex=True)
    df_method['seq2_clean'] = df_method['seq2_id'].str.replace(r'/\d+-\d+', '', regex=True)

    # Merge
    df_merged = pd.merge(
        df_tmalign[['seq1_id', 'seq2_id', 'tm_score']],
        df_method[['seq1_clean', 'seq2_clean', 'tm_score']],
        left_on=['seq1_id', 'seq2_id'],
        right_on=['seq1_clean', 'seq2_clean'],
        suffixes=('_tmalign', '_method')
    )

    print(f"\nMerged {len(df_merged):,} pairs for comparison")

    # Compute and display statistics
    stats_dict = compute_detailed_statistics(df_merged, 'tm_score_method', 'tm_score_tmalign', config['name'])
    print_statistics(stats_dict, verbose=args.verbose)
    
    # Save statistics to JSON if requested
    if args.save_stats:
        import json
        stats_path = Path(args.save_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj
        
        with open(stats_path, 'w') as f:
            json.dump(convert_to_serializable(stats_dict), f, indent=2)
        print(f"Statistics saved to: {stats_path}")

    # Output
    output_dir = Path(args.output_dir or f'figures/{args.method}')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_density_scatter(df_merged, 'tm_score_method', 'tm_score_tmalign', config['name'], output_dir / 'density_scatter')
    print(f"\nDensity plot saved to: {output_dir}/density_scatter.png")


if __name__ == "__main__":
    main()
