#!/bin/bash
# AWS Credentials Setup Script

echo "🔐 AWS Credentials Setup"
echo "========================="
echo ""

echo "Please enter your AWS credentials from the AWS console:"
echo ""

# Get Access Key ID
read -p "Enter your AWS Access Key ID (starts with AKIA): " AWS_ACCESS_KEY_ID
echo ""

# Get Secret Access Key (hidden input)
read -s -p "Enter your AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo ""
echo ""

# Set AWS Region
AWS_DEFAULT_REGION="us-east-1"
echo "Using AWS Region: $AWS_DEFAULT_REGION"
echo ""

# Export environment variables
export AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION"

# Add to shell profile for persistence
SHELL_CONFIG=""
if [ -f ~/.bashrc ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f ~/.bash_profile ]; then
    SHELL_CONFIG="$HOME/.bash_profile"
elif [ -f ~/.zshrc ]; then
    SHELL_CONFIG="$HOME/.zshrc"
fi

if [ -n "$SHELL_CONFIG" ]; then
    echo ""
    echo "Adding AWS credentials to $SHELL_CONFIG for future sessions..."
    
    # Remove existing AWS credentials
    sed -i '/export AWS_ACCESS_KEY_ID=/d' "$SHELL_CONFIG"
    sed -i '/export AWS_SECRET_ACCESS_KEY=/d' "$SHELL_CONFIG"
    sed -i '/export AWS_DEFAULT_REGION=/d' "$SHELL_CONFIG"
    
    # Add new credentials
    echo "" >> "$SHELL_CONFIG"
    echo "# AWS Credentials for S3 Upload" >> "$SHELL_CONFIG"
    echo "export AWS_ACCESS_KEY_ID=\"$AWS_ACCESS_KEY_ID\"" >> "$SHELL_CONFIG"
    echo "export AWS_SECRET_ACCESS_KEY=\"$AWS_SECRET_ACCESS_KEY\"" >> "$SHELL_CONFIG"
    echo "export AWS_DEFAULT_REGION=\"$AWS_DEFAULT_REGION\"" >> "$SHELL_CONFIG"
    
    echo "✓ Credentials saved to $SHELL_CONFIG"
fi

echo ""
echo "🧪 Testing AWS connection..."
echo ""

# Test AWS credentials
if command -v aws &> /dev/null; then
    echo "Testing with AWS CLI..."
    aws s3 ls s3://spatialdino/ --region us-east-1
    if [ $? -eq 0 ]; then
        echo "✅ AWS credentials working! Can access S3 bucket."
    else
        echo "❌ Could not access S3 bucket. Please check your credentials."
    fi
else
    echo "AWS CLI not found. Will test with Python script..."
    python3 -c "
import boto3
try:
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.head_bucket(Bucket='spatialdino')
    print('✅ AWS credentials working! Can access S3 bucket.')
except Exception as e:
    print(f'❌ Error: {e}')
    print('Please check your credentials and bucket permissions.')
"
fi

echo ""
echo "🚀 Ready to proceed with S3 upload setup!"
echo "Next step: Run './run_s3_upload.sh setup'"
echo ""
