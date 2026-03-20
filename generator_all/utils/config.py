"""
Configuration management for AWS S3 and other settings.
"""
import os
from typing import Optional
from dotenv import load_dotenv


class S3Config:
    """AWS S3 Configuration"""
    
    def __init__(self):
        # AWS Credentials - can be set via environment variables or AWS credentials file
        load_dotenv()
        self.aws_access_key_id: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region: str = os.getenv('AWS_REGION', 'us-east-2')
        
        # S3 Bucket Configuration
        self.bucket_name: str = os.getenv('S3_BUCKET_NAME', 'generator-app-files')
        
        # File storage settings
        self.upload_folder: str = 'generated-files'  # Prefix/folder in S3
        self.presigned_url_expiration: int = 3600  # URL expires in 1 hour (3600 seconds)
        
        # File retention (optional - for cleanup jobs)
        self.file_retention_days: int = int(os.getenv('FILE_RETENTION_DAYS', '7'))
    
    def validate(self) -> bool:
        """Validate that required configuration is present"""
        if not self.bucket_name :
            raise ValueError("S3_BUCKET_NAME environment variable must be set")
        return True


# Global config instance
s3_config = S3Config()
