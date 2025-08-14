#!/bin/bash
# Simple wrapper script for AWS CLI operations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_CLI="$SCRIPT_DIR/aws_cli.py"

# Function to show usage
show_usage() {
    echo "Cell Interactome AWS Manager"
    echo "=============================="
    echo ""
    echo "Quick Commands:"
    echo "  $0 setup                          - Setup AWS credentials"
    echo "  $0 upload-data [--dry-run]        - Upload datasets to S3"
    echo "  $0 upload-model MODEL_DIR NAME    - Upload model weights"
    echo "  $0 download-model NAME DIR        - Download model weights"
    echo "  $0 verify                         - Verify uploaded data"
    echo "  $0 status                         - Show upload progress"
    echo "  $0 list-models                    - List available models"
    echo "  $0 storage-usage                  - Show S3 storage usage"
    echo ""
    echo "Advanced Commands:"
    echo "  $0 cli [ARGS...]                  - Run full CLI with arguments"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 upload-data --dry-run"
    echo "  $0 upload-model ./models/backbone.pth vits8-pretrained"
    echo "  $0 download-model vits8-pretrained ./downloaded_models"
    echo "  $0 cli data download dataset_part1/dataset1 ./local_data"
}

# Change to script directory
cd "$SCRIPT_DIR"

case "${1:-help}" in
    "setup")
        echo "Setting up AWS credentials..."
        python "$AWS_CLI" setup
        ;;
    
    "upload-data")
        shift
        echo "Uploading datasets to S3..."
        python "$AWS_CLI" data upload "$@"
        ;;
    
    "upload-model")
        if [ $# -lt 3 ]; then
            echo "Usage: $0 upload-model MODEL_DIR MODEL_NAME"
            exit 1
        fi
        model_dir="$2"
        model_name="$3"
        echo "Uploading model: $model_name"
        python "$AWS_CLI" models upload "$model_dir" "$model_name"
        ;;
    
    "download-model")
        if [ $# -lt 3 ]; then
            echo "Usage: $0 download-model MODEL_NAME DOWNLOAD_DIR"
            exit 1
        fi
        model_name="$2"
        download_dir="$3"
        echo "Downloading model: $model_name"
        python "$AWS_CLI" models download "$model_name" "$download_dir"
        ;;
    
    "verify")
        echo "Verifying uploaded data..."
        python "$AWS_CLI" data verify
        ;;
    
    "status")
        python "$AWS_CLI" data status
        ;;
    
    "list-models")
        python "$AWS_CLI" models list
        ;;
    
    "storage-usage")
        python "$AWS_CLI" storage usage
        ;;
    
    "cli")
        shift
        python "$AWS_CLI" "$@"
        ;;
    
    "help"|*)
        show_usage
        ;;
esac
