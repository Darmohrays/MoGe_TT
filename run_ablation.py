#!/usr/bin/env python3
"""
Script to run eval_baseline.py with multiple TTT config files.

Usage:
    python run_ttt_configs.py /path/to/ttt-configs /path/to/save-folder
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run eval_baseline.py with multiple TTT config files"
    )
    parser.add_argument(
        "--ttt_configs_path",
        type=str,
        help="Path to directory containing TTT config JSON files"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Parent directory for saving output files"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    ttt_configs_dir = Path(args.ttt_configs_path)
    save_dir = Path(args.save_path)
    
    # Validate inputs
    if not ttt_configs_dir.exists():
        print(f"Error: TTT configs directory does not exist: {ttt_configs_dir}")
        sys.exit(1)
    
    if not ttt_configs_dir.is_dir():
        print(f"Error: TTT configs path is not a directory: {ttt_configs_dir}")
        sys.exit(1)
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in the TTT configs directory
    json_files = list(ttt_configs_dir.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {ttt_configs_dir}")
        sys.exit(0)
    
    print(f"Found {len(json_files)} TTT config file(s)")
    print(f"Output will be saved to: {save_dir}")
    print("-" * 80)
    
    # Run command for each config file
    failed_runs = []
    
    for i, config_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {config_file.name}")
        
        # Construct output path: save_dir / config_filename
        output_path = save_dir / config_file.name
        
        # Build the command
        cmd = [
            "python", "moge/scripts/eval_baseline.py",
            "--baseline", "baselines/test_time_train.py",
            "--config", "configs/eval/low_score_moge2_vits.json",
            "--output", str(output_path),
            "--pretrained", "Ruicheng/moge-2-vits-normal",
            "--resolution_level", "9",
            "--version", "v2",
            "--num_tokens", "1500",
            "--dump_pred",
            "--dump_gt",
            "--ttt-config", str(config_file)
        ]
        
        print(f"Running command:")
        print(" ".join(cmd))
        print()
        
        # Execute the command
        try:
            result = subprocess.run(cmd, check=True, text=True)
            print(f"✓ Successfully completed: {config_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {config_file.name}")
            print(f"  Error code: {e.returncode}")
            failed_runs.append(config_file.name)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total configs processed: {len(json_files)}")
    print(f"Successful: {len(json_files) - len(failed_runs)}")
    print(f"Failed: {len(failed_runs)}")
    
    if failed_runs:
        print("\nFailed runs:")
        for name in failed_runs:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\nAll runs completed successfully!")


if __name__ == "__main__":
    main()