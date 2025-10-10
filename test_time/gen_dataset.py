import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def load_metrics(metrics_path):
    """Load metrics from the evaluation results path."""
    # Assuming metrics are stored as CSV or pickle files
    # You may need to adjust this based on your actual file format
    try:
        if (metrics_path / "metrics.csv").exists():
            return pd.read_csv(metrics_path / "metrics.csv", index_col=0)
        elif (metrics_path / "metrics.pkl").exists():
            return pd.read_pickle(metrics_path / "metrics.pkl")
        else:
            # Try to find any CSV or pickle file in the directory
            csv_files = list(metrics_path.glob("*.csv"))
            pkl_files = list(metrics_path.glob("*.pkl"))
            
            if csv_files:
                return pd.read_csv(csv_files[0], index_col=0)
            elif pkl_files:
                return pd.read_pickle(pkl_files[0])
            else:
                raise FileNotFoundError(f"No metrics file found in {metrics_path}")
    except Exception as e:
        print(f"Error loading metrics from {metrics_path}: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset samples based on evaluation metrics and copy low-scoring samples to a new directory."
    )
    
    parser.add_argument(
        "--dump-path",
        type=str,
        default="/media/sviatoslav/nvme0n1p2/University/repos/MoGe/eval_output/moge2_vits/all_benchmarks_dump",
        help="Path to the evaluation results dump directory"
    )
    
    parser.add_argument(
        "--full-dataset-path",
        type=str,
        default="/media/sviatoslav/drive4tb/datasets/monocular_depth/monocular_geometry_evaluation",
        help="Path to the full dataset directory"
    )
    
    parser.add_argument(
        "--new-dataset-path",
        type=str,
        default="/media/sviatoslav/drive4tb/datasets/monocular_depth/low_score_moge2_vits",
        help="Path where filtered dataset will be saved"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold value for filtering samples (default: 0.5)"
    )
    
    parser.add_argument(
        "--threshold-column",
        type=str,
        default="depth_scale_invariant_delta1",
        help="Column name to use for threshold filtering (default: depth_scale_invariant_delta1)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying files"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    dump_path = Path(args.dump_path)
    full_dataset_path = Path(args.full_dataset_path)
    new_dataset_path = Path(args.new_dataset_path)
    
    # Validate input paths
    if not dump_path.exists():
        print(f"Error: Dump path does not exist: {dump_path}")
        return 1
    
    if not full_dataset_path.exists():
        print(f"Error: Full dataset path does not exist: {full_dataset_path}")
        return 1
    
    # Create output directory if it doesn't exist (unless dry run)
    if not args.dry_run and not new_dataset_path.exists():
        new_dataset_path.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            print(f"Created output directory: {new_dataset_path}")
    
    # Get all evaluated dataset paths
    evaluated_datasets_paths = list(dump_path.glob("*"))
    
    if not evaluated_datasets_paths:
        print(f"No evaluation results found in {dump_path}")
        return 1
    
    print(f"Found {len(evaluated_datasets_paths)} evaluated datasets")
    print(f"Using threshold: {args.threshold} on column '{args.threshold_column}'")
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
    
    total_samples = 0
    
    for dataset_eval_results_path in tqdm(evaluated_datasets_paths, desc="Processing datasets"):
        dataset_eval_name = dataset_eval_results_path.name
        
        if args.verbose:
            print(f"\nProcessing dataset: {dataset_eval_name}")
        
        # Load metrics
        dataset_metrics = load_metrics(dataset_eval_results_path)
        
        if dataset_metrics.empty:
            print(f"Warning: No metrics found for {dataset_eval_name}")
            continue
        
        if args.threshold_column not in dataset_metrics.columns:
            print(f"Warning: Column '{args.threshold_column}' not found in {dataset_eval_name}")
            continue
        
        # Filter bad samples
        bad_samples = dataset_metrics[dataset_metrics[args.threshold_column] < args.threshold]
        bad_samples = bad_samples.sort_values(by=args.threshold_column)
        bad_samples_paths = bad_samples.index.to_list()
        
        if args.verbose:
            print(f"Found {len(bad_samples_paths)} samples below threshold")
        
        idx_txt_labels = []
        
        for bad_sample_path in bad_samples_paths:
            new_sample_path = new_dataset_path / dataset_eval_name / bad_sample_path
            old_sample_path = full_dataset_path / dataset_eval_name / bad_sample_path
            
            total_samples += 1
            
            if new_sample_path.exists():
                if args.verbose:
                    print(f"Skipping existing sample: {bad_sample_path}")
                continue
            
            if not old_sample_path.exists():
                print(f"Warning: Source path does not exist: {old_sample_path}")
                continue
            
            if args.dry_run:
                print(f"Would copy: {old_sample_path} -> {new_sample_path}")
            else:
                try:
                    new_sample_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(old_sample_path, new_sample_path)
                    if args.verbose:
                        print(f"Copied: {bad_sample_path}")
                except Exception as e:
                    print(f"Error copying {bad_sample_path}: {e}")
                    continue
            
            idx_txt_labels.append(bad_sample_path + '\n')
        
        # Save index file
        if idx_txt_labels and not args.dry_run:
            index_save_path = new_dataset_path / dataset_eval_name / ".index.txt"
            try:
                with open(index_save_path, "w") as f:
                    f.writelines(idx_txt_labels)
                if args.verbose:
                    print(f"Saved index file: {index_save_path}")
            except Exception as e:
                print(f"Error saving index file: {e}")
        elif idx_txt_labels and args.dry_run:
            print(f"Would create index file with {len(idx_txt_labels)} entries")
    
    print(f"\nProcessing complete!")
    print(f"Total samples {'would be' if args.dry_run else ''} processed: {total_samples}")


if __name__ == "__main__":
    main()