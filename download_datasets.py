#!/usr/bin/env python3
"""
Download TraceLLM datasets from Figshare repository.

Usage:
    python download_datasets.py

"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://figshare.com/ndownloader/files/61771873?private_link=b212efcec17eaa0c70dd"
DATASET_ZIP = "TraceLLM_Datasets.zip"


def download_file(url: str, filename: str) -> None:
    """Download file with progress bar."""
    print(f"Downloading from: {url}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rProgress: {percent}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print("\n✓ Download complete!")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


def check_existing_datasets() -> bool:
    """Check if any datasets already exist."""
    # Check all dataset directories
    dataset_names = ["CCHIT", "CM1_NASA", "EasyClinic_UC_TC", "EasyClinic_UC_ID"]

    for name in dataset_names:
        if Path(f"Datasets/{name}").exists() or Path(f"Embeddings/{name}").exists():
            return True

    return False


def extract_zip(filename: str) -> None:
    """Extract zip file."""
    print(f"\nExtracting {filename}...")
    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("✓ Extraction complete!")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)


def verify_structure() -> None:
    """Verify directory structure."""
    print("\nVerifying structure...")

    required_dirs = ["Datasets", "Embeddings"]
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]

    if missing_dirs:
        print(f"⚠️  Warning: Missing directories: {', '.join(missing_dirs)}")
    else:
        print("✓ Directory structure verified!")

        # List datasets
        datasets = list(Path("Datasets").iterdir())
        if datasets:
            print(f"\nDatasets found: {len(datasets)}")
            for ds in datasets:
                if ds.is_dir():
                    print(f"  - {ds.name}")
        else:
            print("\n⚠️  No datasets found in Datasets/")


def main():
    """Main download and setup function."""
    print("=" * 50)
    print("TraceLLM Dataset Downloader")
    print("=" * 50)
    print()

    # Check if datasets already exist
    if check_existing_datasets():
        print("⚠️  Datasets appear to already exist.")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return

    # Download
    print("\nDownloading datasets...")
    download_file(DATASET_URL, DATASET_ZIP)

    # Extract
    extract_zip(DATASET_ZIP)

    # Cleanup
    print("\nCleaning up...")
    os.remove(DATASET_ZIP)
    print("✓ Temporary files removed")

    # Verify
    verify_structure()

    print("\n" + "=" * 50)
    print("Setup complete! You can now run experiments.")
    print("=" * 50)


if __name__ == "__main__":
    main()
