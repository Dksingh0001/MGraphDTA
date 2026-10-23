"""Build an isolated versioned Davis dataset with author-scaled atom features."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.scaled_features import (
    DEFAULT_DATASET_DIR,
    DEFAULT_FEATURE_COMPARISON_DIR,
    DEFAULT_LIGANDS_FILE,
    DEFAULT_PROCESSED_DIR,
    build_scaled_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_FEATURE_COMPARISON_DIR)
    parser.add_argument("--ligands-file", type=Path, default=DEFAULT_LIGANDS_FILE)
    args = parser.parse_args()
    manifest = build_scaled_dataset(
        args.output_dir,
        processed_dir=args.processed_dir,
        comparison_dir=args.comparison_dir,
        ligands_file=args.ligands_file,
    )
    print(f"Created scaled Davis dataset: {args.output_dir}")
    print(f"Ligands: {manifest['ligand_count']}; atoms: {manifest['atom_count']}")
    for name, metadata in manifest["files"].items():
        print(f"{name}: SHA-256 {metadata['sha256']}")


if __name__ == "__main__":
    main()
