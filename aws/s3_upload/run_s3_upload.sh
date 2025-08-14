#!/bin/bash
# Simple script to run S3 upload with common options

set -e

echo "=========================================="
echo "S3 Dataset Upload Tool"
echo "=========================================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup     - Setup environment and test configuration"
    echo "  dry-run   - Show what would be uploaded without uploading"
    echo "  upload    - Start the upload process"
    echo "  resume    - Resume a previously interrupted upload"
    echo "  verify    - Verify uploaded files integrity"
    echo "  status    - Show current upload status"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 dry-run"
    echo "  $0 upload"
    echo "  $0 verify"
}

# Change to script directory
cd "$(dirname "$0")"

case "${1:-help}" in
    "setup")
        echo "Setting up S3 upload environment..."
        python setup_s3_upload.py
        ;;
    
    "dry-run")
        echo "Running dry-run to show what would be uploaded..."
        python s3_uploader.py --dry-run
        ;;
    
    "upload")
        echo "Starting upload process..."
        echo "Note: You can safely interrupt with Ctrl+C and resume later"
        python s3_uploader.py
        ;;
    
    "resume")
        echo "Resuming upload process..."
        python s3_uploader.py --resume
        ;;
    
    "verify")
        echo "Verifying uploaded files..."
        python s3_verifier.py
        ;;
    
    "status")
        echo "Checking upload status..."
        if [ -f "s3_upload_progress.json" ]; then
            echo "Progress file found. Checking status..."
            python -c "
import json
with open('s3_upload_progress.json', 'r') as f:
    data = json.load(f)
    files = data.get('files', [])
    uploaded = sum(1 for f in files if f.get('uploaded', False))
    total = len(files)
    print(f'Upload progress: {uploaded}/{total} files ({uploaded/total*100:.1f}%)')
"
        else
            echo "No progress file found. Upload hasn't started yet."
        fi
        ;;
    
    "help"|*)
        show_usage
        ;;
esac
