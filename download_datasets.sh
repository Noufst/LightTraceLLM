#!/bin/bash
# Download TraceLLM datasets from Figshare
#
# Usage: ./download_datasets.sh
#
# For double-blind review: Use the private/anonymous link from Figshare
# After acceptance: Update to the permanent public Figshare DOI link

set -e  # Exit on error

echo "================================================"
echo "TraceLLM Dataset Downloader"
echo "================================================"
echo ""

DATASET_URL="https://figshare.com/ndownloader/files/61771873?private_link=b212efcec17eaa0c70dd"
DATASET_ZIP="TraceLLM_Datasets.zip"

echo "Checking if datasets already exist..."
# Check if any of the datasets exist
if [ -d "Datasets/CCHIT" ] || [ -d "Datasets/CM1_NASA" ] || \
   [ -d "Datasets/EasyClinic_UC_TC" ] || [ -d "Datasets/EasyClinic_UC_ID" ] || \
   [ -d "Embeddings/CCHIT" ] || [ -d "Embeddings/CM1_NASA" ] || \
   [ -d "Embeddings/EasyClinic_UC_TC" ] || [ -d "Embeddings/EasyClinic_UC_ID" ]; then
    echo "⚠️  Datasets appear to already exist."
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

echo ""
echo "Downloading datasets..."
echo "Source: $DATASET_URL"
echo ""

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    wget -O "$DATASET_ZIP" "$DATASET_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$DATASET_ZIP" "$DATASET_URL"
else
    echo "❌ Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

echo ""
echo "Extracting datasets..."
unzip -q "$DATASET_ZIP"

echo ""
echo "Cleaning up..."
rm "$DATASET_ZIP"

echo ""
echo "✅ Datasets downloaded successfully!"
echo ""
echo "Verifying structure..."
if [ -d "Datasets" ] && [ -d "Embeddings" ]; then
    echo "✅ Directory structure verified"
    echo ""
    echo "Datasets found:"
    ls -d Datasets/*/ 2>/dev/null || echo "  (none yet)"
else
    echo "⚠️  Warning: Expected directory structure not found."
    echo "Please verify the downloaded files manually."
fi

echo ""
echo "================================================"
echo "Setup complete! You can now run experiments."
echo "================================================"
