# AWS Integration for Cell Interactome

This directory contains all AWS S3 integration tools for managing datasets and model weights in the cloud.

## 🗂️ Directory Structure

```
aws/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── aws_cli.py                   # Comprehensive CLI tool
├── aws_manager.sh               # Simple wrapper script
└── s3_upload/                   # Core upload functionality
    ├── __init__.py              # Package initialization
    ├── s3_upload_config.py      # Configuration settings
    ├── s3_uploader.py           # Dataset upload engine
    ├── s3_verifier.py           # Data integrity verification
    ├── model_uploader.py        # Model weights upload
    ├── s3_upload_summary.py     # Status and progress display
    ├── setup_s3_upload.py       # Environment setup
    ├── setup_aws_credentials.sh # Credential configuration
    ├── run_s3_upload.sh         # Legacy upload script
    ├── requirements_s3.txt      # Core dependencies
    └── README_S3_Upload.md      # Detailed documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup AWS Credentials

```bash
./aws_manager.sh setup
```

### 3. Upload Data and Models

```bash
# Upload datasets
./aws_manager.sh upload-data

# Upload model weights
./aws_manager.sh upload-model ../models my-model-name
```

## 🛠️ Available Tools

### Simple Manager (`aws_manager.sh`)

Easy-to-use wrapper for common operations:

- `setup` - Configure AWS credentials
- `upload-data` - Upload datasets
- `upload-model` - Upload model weights
- `download-model` - Download models
- `status` - Check progress
- `verify` - Verify integrity

### Comprehensive CLI (`aws_cli.py`)

Full-featured command-line interface:

- **Data operations**: `data upload`, `data download`, `data verify`
- **Model operations**: `models upload`, `models download`, `models list`
- **Storage management**: `storage list`, `storage usage`, `storage cleanup`

### Legacy Scripts (`s3_upload/`)

Original upload tools for advanced users:

- Direct Python scripts for custom workflows
- Detailed configuration options
- Advanced features like bandwidth limiting

## 📊 Data Organization

### S3 Bucket Structure

```
s3://spatialdino/dataset_part1/
├── dataset1/          # RAID1 data (870 files, 817GB)
├── dataset2/          # RAID2 data (780 files, 732GB)
├── dataset3/          # RAID3 data (770 files, 723GB)
└── models/           # Model weights and configurations
    ├── backbone/
    ├── checkpoints/
    └── custom-models/
```

### Local Directory Mapping

- `/raid1/shared_image_recog_ml/llsm_3d_ds_auto_crop` → `dataset1/`
- `/raid2/shared_image_recog_ml/llsm_3d_ds_auto_crop` → `dataset2/`
- `/raid3/shared_image_recog_ml/llsm_3d_ds_auto_crop` → `dataset3/`

## ⚙️ Configuration

### Environment Variables

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Optional overrides
export S3_BUCKET_NAME=spatialdino
export MAX_CONCURRENCY=4
export CHUNK_SIZE_MB=100
```

### Config File (`s3_upload/s3_upload_config.py`)

- S3 bucket and region settings
- Dataset directory mappings
- Upload performance tuning
- Retry and bandwidth settings

## 🔧 Advanced Usage

### Custom Upload Workflows

```python
from aws.s3_upload import S3Uploader, ModelUploader

# Upload datasets
uploader = S3Uploader()
uploader.run()

# Upload models
model_uploader = ModelUploader()
model_uploader.upload_model("./models", "my-model")
```

### Monitoring and Verification

```bash
# Check upload progress
python aws_cli.py data status

# Verify data integrity
python aws_cli.py data verify

# Monitor S3 usage
python aws_cli.py storage usage
```

## 📋 Performance

- **Upload speed**: 50-200 MB/s (depends on network and concurrency)
- **Total dataset**: ~2.3 TB across 3 RAID systems
- **Estimated time**: 2-4 hours for full upload
- **Resume capability**: Automatic continuation of interrupted uploads

## 🆘 Troubleshooting

### Common Issues

**AWS Credentials Not Found**

```bash
./aws_manager.sh setup
```

**Upload Interrupted**

```bash
# Uploads automatically resume on restart
./aws_manager.sh upload-data
```

**Verification Failures**

```bash
# Check specific dataset
python aws_cli.py data verify

# Re-upload failed files
python aws_cli.py data upload
```

### Logs and Debug Info

- Upload logs: `s3_upload.log`
- Progress tracking: `s3_upload_progress.json`
- Model upload logs: `model_upload.log`

## 🔒 Security Notes

- AWS credentials are stored securely in environment variables
- S3 bucket permissions should be configured appropriately
- Use IAM roles with minimal required permissions
- Enable MFA for AWS account security

---

For detailed technical documentation, see `s3_upload/README_S3_Upload.md`
