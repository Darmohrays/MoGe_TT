#!/usr/bin/env python3
"""
Script to analyze and visualize metrics from model evaluation dumps.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np


def _metric_columns(kind):
    """
    Return (depth_cols, points_cols) for a given metric kind: 'delta1' or 'rel'.
    """
    if kind == "delta1":
        depth_cols = ["depth_metric_delta1",
                      "depth_scale_invariant_delta1",
                      "depth_affine_invariant_delta1"]
        points_cols = ["points_metric_delta1",
                       "points_scale_invariant_delta1",
                       "points_affine_invariant_delta1"]
    elif kind == "rel":
        depth_cols = ["depth_metric_rel",
                      "depth_scale_invariant_rel",
                      "depth_affine_invariant_rel"]
        points_cols = ["points_metric_rel",
                       "points_scale_invariant_rel",
                       "points_affine_invariant_rel"]
    else:
        raise ValueError(f"Unknown metric kind: {kind}")
    return depth_cols, points_cols


def flatten_dict(nested_dict, separator='_'):
    """
    Flatten a nested dictionary by combining keys with a separator.
    
    Args:
        nested_dict (dict): The nested dictionary to flatten
        separator (str): The separator to use between parent and child keys (default: '_')
    
    Returns:
        dict: Flattened dictionary with combined keys
    """
    flattened = {}
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # If value is a dictionary, recursively flatten it
            for sub_key, sub_value in value.items():
                flattened_key = f"{key}{separator}{sub_key}"
                flattened[flattened_key] = sub_value
        else:
            # If value is not a dictionary, keep it as is
            flattened[key] = value
    
    return flattened


def load_metrics(path, idx_col="sample_id"):
    """
    Load all metrics.json files from a directory recursively.
    
    Args:
        path: Path to the directory containing metrics files
        idx_col: Column to use as index (default: "sample_id")
    
    Returns:
        DataFrame with all metrics
    """
    # Find all metrics.json files recursively in all subdirectories
    metrics_files = list(path.rglob("**/metrics.json"))
    samples_metrics = []
    
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            sample_metrics = json.load(f)
        sample_metrics_flattened = flatten_dict(sample_metrics)
        
        # Get the relative path from the base path to the metrics file's parent directory
        sample_path = metrics_file.parents[1].relative_to(path)
        sample_id = str(sample_path)
        sample_metrics_flattened['sample_id'] = sample_id
        samples_metrics.append(sample_metrics_flattened)
    
    df = pd.DataFrame(samples_metrics)
    if idx_col is not None:
        df = df.set_index(idx_col)
    return df


def save_loss_plot(dump_tt_path, save_path, name, smooth=True, smooth_method="ema", 
                   ema_alpha=0.1, ma_window=25, num_samples=20):
    """
    Create and save loss plot for a given dump path.
    
    Args:
        dump_tt_path: Path to the dump directory
        save_path: Path where to save the plot
        name: Name for the saved file
        smooth: Whether to apply smoothing
        smooth_method: Smoothing method ("ema" or "ma")
        ema_alpha: EMA alpha parameter
        ma_window: Moving average window size
        num_samples: Number of random samples to plot
    """
    losses_logs_paths = list(dump_tt_path.glob("**/losses_logs.csv"))
    if len(losses_logs_paths) > num_samples:
        losses_logs_paths = random.choices(losses_logs_paths, k=num_samples)
    
    # Collect all metric values first to normalize the colormap
    metric_values = []
    plot_data = []
    
    for losses_log_path in losses_logs_paths:
        try:
            df = pd.read_csv(losses_log_path, index_col=0)
            metrics_path = losses_log_path.parent / "metrics.json"
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            metric_value = metrics['points_scale_invariant']['delta1']
            
            plot_data.append((df, metric_value))
            metric_values.append(metric_value)
        except (KeyError, FileNotFoundError):
            continue
    
    if not plot_data:
        print(f"Warning: No valid loss logs found in {dump_tt_path}")
        return
    
    # Create colormap and normalization
    cmap = cm.viridis
    norm = plt.Normalize(vmin=min(metric_values), vmax=max(metric_values))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for df, metric_value in plot_data:
        color = cmap(norm(metric_value))
        
        # Get the loss as numeric (in case of stray strings)
        loss = pd.to_numeric(df["loss"], errors="coerce")
        
        # Apply smoothing if requested
        if smooth:
            if smooth_method == "ema":
                y = loss.ewm(alpha=ema_alpha, adjust=False).mean()
                ylabel = f"Loss (EMA α={ema_alpha})"
            else:
                y = loss.rolling(window=ma_window, min_periods=1).mean()
                ylabel = f"Loss (MA w={ma_window})"
        else:
            y = loss
            ylabel = "Loss"
        
        ax.plot(df.index, y, color=color, alpha=0.9, linewidth=1.5)
    
    ax.set_xlabel("Iter")
    ax.set_ylabel(ylabel)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Delta1 Metric', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(save_path / f"{name}_loss_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_scatter_plot(dump_metrics, dump_tt_metrics, save_path, name, metric_kind="delta1"):
    """
    Create and save scatter plots comparing metrics for the given kind ('delta1' or 'rel').
    """
    depth_cols, points_cols = _metric_columns(metric_kind)

    f, ax = plt.subplots(3, 2, figsize=(13, 17))

    for i, col_name in enumerate(depth_cols):
        # Align indices between baseline and TT
        common_idx = dump_tt_metrics.index.intersection(dump_metrics.index)
        v1_data = dump_metrics.loc[common_idx, col_name]
        v1_tt_data = dump_tt_metrics.loc[common_idx, col_name]

        ax[i, 0].scatter(v1_data, v1_tt_data, alpha=0.6)
        ax[i, 0].set_xlabel("Original")
        ax[i, 0].set_ylabel("With TT")

        min_value = np.nanmin([v1_data.min(), v1_tt_data.min()])
        max_value = np.nanmax([v1_data.max(), v1_tt_data.max()])
        line = np.linspace(min_value, max_value, 50)

        ax[i, 0].plot(line, line, 'g', alpha=0.5)
        ax[i, 0].set_title(col_name)
        ax[i, 0].grid(alpha=0.3)

    for i, col_name in enumerate(points_cols):
        common_idx = dump_tt_metrics.index.intersection(dump_metrics.index)
        v1_data = dump_metrics.loc[common_idx, col_name]
        v1_tt_data = dump_tt_metrics.loc[common_idx, col_name]

        ax[i, 1].scatter(v1_data, v1_tt_data, alpha=0.6)
        ax[i, 1].set_xlabel("Original")
        ax[i, 1].set_ylabel("With TT")

        min_value = np.nanmin([v1_data.min(), v1_tt_data.min()])
        max_value = np.nanmax([v1_data.max(), v1_tt_data.max()])
        line = np.linspace(min_value, max_value, 50)

        ax[i, 1].plot(line, line, 'g', alpha=0.5)
        ax[i, 1].set_title(col_name)
        ax[i, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / f"{name}_scatter_plot_{metric_kind}.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_bar_plot(dump_metrics, dump_tt_metrics, save_path, name, metric_kind="delta1"):
    """
    Create and save bar plots showing better/worse counts for the given kind ('delta1' or 'rel').
    For delta1: higher is better; for rel: lower is better.
    """
    depth_cols, points_cols = _metric_columns(metric_kind)
    higher_is_better = (metric_kind == "delta1")

    f, ax = plt.subplots(3, 2, figsize=(13, 17))

    for i, col_name in enumerate(depth_cols):
        common_idx = dump_tt_metrics.index.intersection(dump_metrics.index)
        v1_data = dump_metrics.loc[common_idx, col_name]
        v1_tt_data = dump_tt_metrics.loc[common_idx, col_name]

        # diff > 0 means TT better if higher_is_better; invert for rel
        diff = (v1_tt_data - v1_data) if higher_is_better else (v1_data - v1_tt_data)
        better = (diff > 0).astype(int)

        counts = better.value_counts().reindex([0, 1]).fillna(0).astype(int)

        ax[i, 0].bar([0, 1], counts.values, color=['red', 'green'])
        ax[i, 0].set_title(col_name)
        ax[i, 0].set_xlabel('Performance')
        ax[i, 0].set_ylabel('Count')
        ax[i, 0].set_xticks([0, 1])
        ax[i, 0].set_xticklabels(['Worse', 'Better'])
        ax[i, 0].grid(axis='y', alpha=0.3)

    for i, col_name in enumerate(points_cols):
        common_idx = dump_tt_metrics.index.intersection(dump_metrics.index)
        v1_data = dump_metrics.loc[common_idx, col_name]
        v1_tt_data = dump_tt_metrics.loc[common_idx, col_name]

        diff = (v1_tt_data - v1_data) if higher_is_better else (v1_data - v1_tt_data)
        better = (diff > 0).astype(int)

        counts = better.value_counts().reindex([0, 1]).fillna(0).astype(int)

        ax[i, 1].bar([0, 1], counts.values, color=['red', 'green'])
        ax[i, 1].set_title(col_name)
        ax[i, 1].set_xlabel('Performance')
        ax[i, 1].set_ylabel('Count')
        ax[i, 1].set_xticks([0, 1])
        ax[i, 1].set_xticklabels(['Worse', 'Better'])
        ax[i, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / f"{name}_bar_plot_{metric_kind}.png", dpi=300, bbox_inches='tight')
    plt.close()



def save_comparison_plot(dump_metrics, all_dump_tt_metrics, all_names, save_path):
    """
    Create and save comparison and heatmap plot.
    
    Args:
        dump_metrics: Baseline metrics DataFrame
        all_dump_tt_metrics: List of test-time metrics DataFrames
        all_names: List of names for each test-time dump
        save_path: Path where to save the plot
    """
    # Define the metric columns (delta1 metrics)
    depth_cols = ["depth_metric_delta1", 
                  "depth_scale_invariant_delta1",
                  "depth_affine_invariant_delta1"]
    points_cols = ["points_metric_delta1", 
                   "points_scale_invariant_delta1",
                   "points_affine_invariant_delta1"]
    
    all_cols = depth_cols + points_cols
    
    # Define whether higher or lower values are better for each metric
    metric_direction = {col: True for col in all_cols}  # All delta1 metrics: higher is better
    
    # Calculate for all dumps
    all_grand_means = {'Original': {}}
    for col in all_cols:
        all_grand_means['Original'][col] = dump_metrics[col].mean()
    
    for name, dump_tt_metrics in zip(all_names, all_dump_tt_metrics):
        all_grand_means[name] = {}
        for col in all_cols:
            all_grand_means[name][col] = dump_tt_metrics[col].mean()
    
    # Create DataFrame for visualization
    heatmap_data = pd.DataFrame(all_grand_means).T
    
    # Create the visualization
    num_methods = len(all_grand_means)
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Define distinguishable colors for each method
    colors = plt.cm.tab10(np.linspace(0, 1, num_methods))
    method_colors = {method: colors[i] for i, method in enumerate(all_grand_means.keys())}
    
    # Plot 1: Side-by-side bars
    bar_width = 0.8 / num_methods
    for i, metric in enumerate(all_cols):
        for j, method in enumerate(all_grand_means.keys()):
            value = all_grand_means[method][metric]
            color = method_colors[method]
            offset = (j - num_methods/2 + 0.5) * bar_width
            axes[0].barh(i + offset, value, height=bar_width, 
                        color=color, alpha=0.8, label=method if i == 0 else "")
            
            # Add value labels
            axes[0].text(value + value*0.01, i + offset, f'{value:.4f}', 
                        va='center', fontsize=8)
    
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    
    # Customize first plot
    axes[0].set_yticks(range(len(all_cols)))
    axes[0].set_yticklabels([f"{col} ({'↑' if metric_direction[col] else '↓'})" 
                             for col in all_cols])
    axes[0].set_xlabel('Metric Values')
    # axes[0].set_title('Comparison\n↑=Higher is better, ↓=Lower is better')
    # axes[0].legend(loc='best')
    # axes[0].grid(axis='x', alpha=0.3)
    # axes[0].invert_yaxis()
    axes[0].set_title('Comparison\n↑=Higher is better, ↓=Lower is better')

    # Legend below the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    # de-duplicate legend entries, preserving order
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l); h2.append(h); l2.append(l)

    axes[0].legend(
        h2, l2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),  # push below axes[0]
        ncol=min(len(h2), 2),
        frameon=False,
        borderaxespad=0.0
    )

    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()

    # Layout: leave extra bottom room for the legend under axes[0]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    
    # Plot 2: Heatmap with performance coloring relative to baseline
    heatmap_for_colors = heatmap_data.copy()
    
    # Get baseline values
    baseline_values = heatmap_data.loc['Original']
    
    for metric in all_cols:
        baseline_val = baseline_values[metric]
        higher_is_better = metric_direction[metric]
        
        for method in heatmap_data.index:
            value = heatmap_data.loc[method, metric]
            
            if method == 'Original':
                # Baseline is always neutral (0.5)
                heatmap_for_colors.loc[method, metric] = 0.5
            else:
                # Calculate relative performance
                if higher_is_better:
                    # For delta1: higher is better
                    # If value > baseline: green (> 0.5)
                    # If value < baseline: red (< 0.5)
                    diff = value - baseline_val
                else:
                    # For rel: lower is better
                    # If value < baseline: green (> 0.5)
                    # If value > baseline: red (< 0.5)
                    diff = baseline_val - value
                
                # Normalize difference to [0, 1] range
                # Use absolute max difference across all methods for this metric to normalize
                all_diffs = []
                for m in heatmap_data.index:
                    if m != 'Original':
                        if higher_is_better:
                            all_diffs.append(heatmap_data.loc[m, metric] - baseline_val)
                        else:
                            all_diffs.append(baseline_val - heatmap_data.loc[m, metric])
                
                max_abs_diff = max(abs(min(all_diffs)), abs(max(all_diffs))) if all_diffs else 1
                
                if max_abs_diff > 0:
                    # Map to [0, 1] where 0.5 is baseline
                    # Positive diff (better) -> >0.5 (green)
                    # Negative diff (worse) -> <0.5 (red)
                    normalized = 0.5 + (diff / (2 * max_abs_diff))
                    # Clamp to [0, 1]
                    normalized = max(0, min(1, normalized))
                    heatmap_for_colors.loc[method, metric] = normalized
                else:
                    heatmap_for_colors.loc[method, metric] = 0.5
    
    # Create heatmap
    im = axes[1].imshow(heatmap_for_colors.T.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations with actual values
    for i in range(len(all_cols)):
        for j in range(num_methods):
            text = axes[1].text(j, i, f'{heatmap_data.iloc[j, i]:.4f}', 
                               ha="center", va="center", color="black", fontweight='bold',
                               fontsize=8)
    
    # Customize heatmap
    axes[1].set_xticks(range(num_methods))
    axes[1].set_xticklabels(heatmap_data.index, rotation=45, ha='right')
    axes[1].set_yticks(range(len(all_cols)))
    axes[1].set_yticklabels([f"{col} ({'↑' if metric_direction[col] else '↓'})" 
                             for col in all_cols])
    axes[1].set_title('Heatmap Relative to Baseline\n(Green=Better than baseline, Red=Worse than baseline)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], shrink=0.6)
    cbar.set_label('Performance relative to baseline')
    
    plt.tight_layout()
    plt.savefig(save_path / "comparison_plot_delta1.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now create the same plots for rel metrics
    depth_cols_rel = ["depth_metric_rel",
                      "depth_scale_invariant_rel", 
                      "depth_affine_invariant_rel"]
    points_cols_rel = ["points_metric_rel",
                       "points_scale_invariant_rel",
                       "points_affine_invariant_rel"]
    
    all_cols_rel = depth_cols_rel + points_cols_rel
    
    # For rel metrics, lower is better
    metric_direction_rel = {col: False for col in all_cols_rel}
    
    # Calculate for rel metrics
    all_grand_means_rel = {'Original': {}}
    for col in all_cols_rel:
        all_grand_means_rel['Original'][col] = dump_metrics[col].mean()
    
    for name, dump_tt_metrics in zip(all_names, all_dump_tt_metrics):
        all_grand_means_rel[name] = {}
        for col in all_cols_rel:
            all_grand_means_rel[name][col] = dump_tt_metrics[col].mean()
    
    # Create DataFrame for visualization
    heatmap_data_rel = pd.DataFrame(all_grand_means_rel).T
    
    # Create the visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Plot 1: Side-by-side bars
    for i, metric in enumerate(all_cols_rel):
        for j, method in enumerate(all_grand_means_rel.keys()):
            value = all_grand_means_rel[method][metric]
            color = method_colors[method]
            offset = (j - num_methods/2 + 0.5) * bar_width
            axes[0].barh(i + offset, value, height=bar_width, 
                        color=color, alpha=0.8, label=method if i == 0 else "")
            
            # Add value labels
            axes[0].text(value + value*0.01, i + offset, f'{value:.4f}', 
                        va='center', fontsize=8)
    
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    
    # Customize first plot
    axes[0].set_yticks(range(len(all_cols_rel)))
    axes[0].set_yticklabels([f"{col} ({'↑' if metric_direction_rel[col] else '↓'})" 
                             for col in all_cols_rel])
    axes[0].set_xlabel('Metric Values')
    # axes[0].set_title('Comparison (Rel Metrics)\n↑=Higher is better, ↓=Lower is better')
    # axes[0].legend(loc='best')
    # axes[0].grid(axis='x', alpha=0.3)
    # axes[0].invert_yaxis()
    axes[0].set_title('Comparison (Rel Metrics)\n↑=Higher is better, ↓=Lower is better')

    # Legend below the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l); h2.append(h); l2.append(l)

    axes[0].legend(
        h2, l2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(h2), 2),
        frameon=False,
        borderaxespad=0.0
    )

    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    
    # Plot 2: Heatmap with performance coloring relative to baseline
    heatmap_for_colors_rel = heatmap_data_rel.copy()
    
    # Get baseline values
    baseline_values_rel = heatmap_data_rel.loc['Original']
    
    for metric in all_cols_rel:
        baseline_val = baseline_values_rel[metric]
        higher_is_better = metric_direction_rel[metric]
        
        for method in heatmap_data_rel.index:
            value = heatmap_data_rel.loc[method, metric]
            
            if method == 'Original':
                # Baseline is always neutral (0.5)
                heatmap_for_colors_rel.loc[method, metric] = 0.5
            else:
                # Calculate relative performance
                if higher_is_better:
                    # For metrics where higher is better
                    # If value > baseline: green (> 0.5)
                    # If value < baseline: red (< 0.5)
                    diff = value - baseline_val
                else:
                    # For rel: lower is better
                    # If value < baseline: green (> 0.5)
                    # If value > baseline: red (< 0.5)
                    diff = baseline_val - value
                
                # Normalize difference to [0, 1] range
                # Use absolute max difference across all methods for this metric to normalize
                all_diffs = []
                for m in heatmap_data_rel.index:
                    if m != 'Original':
                        if higher_is_better:
                            all_diffs.append(heatmap_data_rel.loc[m, metric] - baseline_val)
                        else:
                            all_diffs.append(baseline_val - heatmap_data_rel.loc[m, metric])
                
                max_abs_diff = max(abs(min(all_diffs)), abs(max(all_diffs))) if all_diffs else 1
                
                if max_abs_diff > 0:
                    # Map to [0, 1] where 0.5 is baseline
                    # Positive diff (better) -> >0.5 (green)
                    # Negative diff (worse) -> <0.5 (red)
                    normalized = 0.5 + (diff / (2 * max_abs_diff))
                    # Clamp to [0, 1]
                    normalized = max(0, min(1, normalized))
                    heatmap_for_colors_rel.loc[method, metric] = normalized
                else:
                    heatmap_for_colors_rel.loc[method, metric] = 0.5
    
    # Create heatmap
    im = axes[1].imshow(heatmap_for_colors_rel.T.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(all_cols_rel)):
        for j in range(num_methods):
            text = axes[1].text(j, i, f'{heatmap_data_rel.iloc[j, i]:.4f}', 
                               ha="center", va="center", color="black", fontweight='bold',
                               fontsize=8)
    
    # Customize heatmap
    axes[1].set_xticks(range(num_methods))
    axes[1].set_xticklabels(heatmap_data_rel.index, rotation=45, ha='right')
    axes[1].set_yticks(range(len(all_cols_rel)))
    axes[1].set_yticklabels([f"{col} ({'↑' if metric_direction_rel[col] else '↓'})" 
                             for col in all_cols_rel])
    axes[1].set_title('Heatmap Relative to Baseline (Rel Metrics)\n(Green=Better than baseline, Red=Worse than baseline)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], shrink=0.6)
    cbar.set_label('Performance relative to baseline')
    
    plt.tight_layout()
    plt.savefig(save_path / "comparison_plot_rel.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_csv(dump_metrics, all_dump_tt_metrics, all_names, save_path, filename="comparison_metrics.csv"):
    """
    Save a CSV where each row is an experiment (baseline + each ablation)
    and each column is a metric value (grand mean over samples).
    The first column is 'experiment' with the experiment name.
    """
    # Metric columns already used elsewhere in the script
    depth_cols_delta1 = [
        "depth_metric_delta1",
        "depth_scale_invariant_delta1",
        "depth_affine_invariant_delta1"
    ]
    points_cols_delta1 = [
        "points_metric_delta1",
        "points_scale_invariant_delta1",
        "points_affine_invariant_delta1"
    ]

    depth_cols_rel = [
        "depth_metric_rel",
        "depth_scale_invariant_rel",
        "depth_affine_invariant_rel"
    ]
    points_cols_rel = [
        "points_metric_rel",
        "points_scale_invariant_rel",
        "points_affine_invariant_rel"
    ]

    # Final column order: delta1 then rel (depth first, then points)
    ordered_cols = (
        depth_cols_delta1 + points_cols_delta1 +
        depth_cols_rel   + points_cols_rel
    )

    # Make sure we only keep columns that actually exist in the data
    existing_cols = [c for c in ordered_cols if c in dump_metrics.columns]

    rows = []

    # Baseline ("Original")
    baseline_row = {"experiment": "Original"}
    for c in existing_cols:
        baseline_row[c] = dump_metrics[c].mean()
    rows.append(baseline_row)

    # Each ablation / TT experiment
    for name, df in zip(all_names, all_dump_tt_metrics):
        row = {"experiment": name}
        for c in existing_cols:
            if c in df.columns:
                row[c] = df[c].mean()
            else:
                row[c] = np.nan
        rows.append(row)

    # Build DataFrame with 'experiment' as the first column
    df_out = pd.DataFrame(rows)
    # Ensure columns order (experiment first)
    df_out = df_out[["experiment"] + existing_cols]

    # Save
    csv_path = save_path / filename
    df_out.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize model evaluation metrics')
    parser.add_argument('--save-path', type=Path, required=True,
                        help='Path where all visualizations should be saved')
    parser.add_argument('--dump-path', type=Path, required=True,
                        help='Path to the baseline results')
    parser.add_argument('--dump-tt-paths', type=Path, nargs='+', required=True,
                        help='List of paths to test-time adaptation results')
    parser.add_argument('--save-scatter-plot', action='store_true',
                        help='Save scatter plots displaying metrics')
    parser.add_argument('--save-loss-plot', action='store_true',
                        help='Save loss plots')
    parser.add_argument('--save-bar-plot', action='store_true',
                        help='Save True/False comparison bar plots')
    parser.add_argument('--save-comparison-plot', action='store_true',
                        help='Save comparison and heatmap plot')
    parser.add_argument('--save-comparison-csv', action='store_true',
                        help='Save a CSV with grand-mean metric values per experiment')

    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    args.save_path.mkdir(parents=True, exist_ok=True)
    
    # Load baseline metrics
    print(f"Loading baseline metrics from {args.dump_path}...")
    dump_metrics = load_metrics(args.dump_path)
    print(f"Loaded {len(dump_metrics)} baseline samples")
    
    # Load all test-time metrics
    all_dump_tt_metrics = []
    all_names = []
    
    for dump_tt_path in args.dump_tt_paths:
        print(f"Loading test-time metrics from {dump_tt_path}...")
        dump_tt_metrics = load_metrics(dump_tt_path)
        print(f"Loaded {len(dump_tt_metrics)} test-time samples")
        all_dump_tt_metrics.append(dump_tt_metrics)
        all_names.append(dump_tt_path.name)
    
    # Generate plots for each dump-tt-path
    for i, (dump_tt_path, dump_tt_metrics, name) in enumerate(zip(args.dump_tt_paths, 
                                                                    all_dump_tt_metrics, 
                                                                    all_names)):
        print(f"\nProcessing {name} ({i+1}/{len(args.dump_tt_paths)})...")
        
        if args.save_scatter_plot:
            print("  Creating scatter plots (delta1 & rel)...")
            save_scatter_plot(dump_metrics, dump_tt_metrics, args.save_path, name, metric_kind="delta1")
            save_scatter_plot(dump_metrics, dump_tt_metrics, args.save_path, name, metric_kind="rel")

        if args.save_bar_plot:
            print("  Creating bar plots (delta1 & rel)...")
            save_bar_plot(dump_metrics, dump_tt_metrics, args.save_path, name, metric_kind="delta1")
            save_bar_plot(dump_metrics, dump_tt_metrics, args.save_path, name, metric_kind="rel")

        if args.save_loss_plot:
            print("  Creating loss plot...")
            save_loss_plot(dump_tt_path, args.save_path, name)
        
    
    # Generate comparison plot with all dumps
    if args.save_comparison_plot:
        print("\nCreating comparison plot for all dumps...")
        save_comparison_plot(dump_metrics, all_dump_tt_metrics, all_names, args.save_path)
    
    # Generate CSV with for all dumps
    if args.save_comparison_csv:
        save_comparison_csv(dump_metrics, all_dump_tt_metrics, all_names, args.save_path)

    print(f"\nAll visualizations saved to {args.save_path}")


if __name__ == "__main__":
    main()