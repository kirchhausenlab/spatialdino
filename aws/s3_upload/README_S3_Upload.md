# S3 Upload Tool for Large Datasets

This tool efficiently uploads hundreds of gigabytes of data from your local servers to AWS S3 with progress tracking, resume capability, and verification.

## Features

- **Parallel uploads** with configurable concurrency
- **Resume capability** - interrupted uploads can be resumed
- **Progress tracking** with detailed logging
- **Bandwidth limiting** to control network usage
- **Multipart uploads** for large files
- **Verification** to ensure data integrity
- **Skip existing files** to avoid re-uploading

## Quick Start

### 1. Setup

```bash
# Install dependencies and check configuration
python setup_s3_upload.py
```

### 2. Configure AWS Credentials

Option A - Environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

Option B - AWS CLI:

```bash
aws configure
```

### 3. Test Configuration

```bash
# See what would be uploaded without actually uploading
python s3_uploader.py --dry-run
```

### 4. Start Upload

```bash
# Start the upload process
python s3_uploader.py
```

### 5. Verify Upload

```bash
# Verify all files uploaded correctly
python s3_verifier.py
```

## Configuration

Edit `s3_upload_config.py` to customize:

- **Dataset mappings**: Which local directories map to which S3 paths
- **Upload settings**: Chunk size, concurrency, bandwidth limits
- **AWS settings**: Bucket name, region, etc.

Default mappings:

- `/raid1/shared_image_recog_ml/llsm_3d_ds_auto_crop` → `s3://spatialdino/dataset_part1/dataset1/`
- `/raid2/shared_image_recog_ml/llsm_3d_ds_auto_crop` → `s3://spatialdino/dataset_part1/dataset2/`
- `/raid3/shared_image_recog_ml/llsm_3d_ds_auto_crop` → `s3://spatialdino/dataset_part1/dataset3/`

## Environment Variables

You can override configuration with environment variables:

```bash
export S3_BUCKET_NAME=spatialdino
export S3_PREFIX=dataset_part1
export MAX_CONCURRENCY=4
export CHUNK_SIZE_MB=100
export MAX_BANDWIDTH_MBPS=100  # Optional bandwidth limit
```

## Resume Uploads

If an upload is interrupted, simply run the uploader again:

```bash
python s3_uploader.py
```

The tool automatically:

- Skips files that are already uploaded
- Resumes incomplete multipart uploads
- Maintains progress in `s3_upload_progress.json`

## Monitoring

The tool provides:

- **Real-time progress bar** with transfer speeds
- **Detailed logging** to `s3_upload.log` and console
- **Resource monitoring** (CPU, memory usage warnings)
- **Per-file status** (success/failure tracking)

## Advanced Usage

### Custom Configuration

Create a custom config file and modify `s3_upload_config.py`:

```python
# Custom dataset mappings
dataset_mappings = {
    "/path/to/your/data1": "custom_dataset1",
    "/path/to/your/data2": "custom_dataset2"
}
```

### Performance Tuning

```bash
# Increase concurrency for faster uploads (use with caution)
export MAX_CONCURRENCY=8

# Increase chunk size for very large files
export CHUNK_SIZE_MB=200

# Limit bandwidth during business hours
export MAX_BANDWIDTH_MBPS=50
```

### Verification Only

```bash
# Check for orphaned files in S3
python s3_verifier.py --orphaned-only

# Full verification of all files
python s3_verifier.py
```

## Troubleshooting

### AWS Credentials Issues

```
Error: AWS credentials not found
```

- Check that AWS credentials are properly configured
- Run `aws s3 ls` to test AWS CLI access

### Permission Issues

```
Error: Access Denied
```

- Ensure your AWS user has S3 permissions for the bucket
- Check bucket policies and IAM permissions

### Network Issues

```
Error: Connection timeout
```

- Check internet connectivity
- Consider reducing `MAX_CONCURRENCY`
- Try setting `MAX_BANDWIDTH_MBPS` to limit transfer rate

### Disk Space Issues

```
Error: No space left on device
```

- The tool doesn't require extra disk space for uploads
- Check that log files aren't filling up disk

### Resume Failed Uploads

```bash
# Clean up incomplete multipart uploads if needed
aws s3api list-multipart-uploads --bucket spatialdino --prefix dataset_part1

# Abort specific upload if necessary
aws s3api abort-multipart-upload --bucket spatialdino --key <key> --upload-id <upload-id>
```

## File Structure

```
cell_interactome/
├── s3_upload_config.py      # Configuration settings
├── s3_uploader.py           # Main upload script
├── s3_verifier.py           # Verification script
├── setup_s3_upload.py       # Setup and testing script
├── requirements_s3.txt      # Python dependencies
├── s3_upload.log           # Upload log (created during upload)
└── s3_upload_progress.json # Progress tracking (created during upload)
```

## Performance Expectations

For typical large file transfers:

- **Throughput**: 50-200 MB/s (depends on network and concurrency)
- **Memory usage**: ~100-500 MB
- **CPU usage**: Low (mostly I/O bound)

Estimated upload times for your datasets:

- 100 GB: ~10-30 minutes
- 500 GB: ~1-2 hours
- 1 TB: ~2-4 hours

## Support

If you encounter issues:

1. Check the log file `s3_upload.log` for detailed error messages
2. Run with `--dry-run` to test configuration
3. Verify AWS credentials and permissions
4. Check network connectivity and S3 service status
