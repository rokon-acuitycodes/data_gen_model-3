"""
AWS S3 Storage Manager for file uploads and downloads.
Provides a reusable interface for storing generated files in S3.
"""
import io
import uuid
import zipfile
from datetime import datetime
from typing import List, Tuple, Optional, BinaryIO
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from .config import s3_config


class S3Manager:
    """
    Manages file uploads, downloads, and ZIP creation with AWS S3.
    
    Usage:
        manager = S3Manager()
        
        # Upload a single file
        url = manager.upload_file(file_data, 'image.png', 'image/png')
        
        # Upload multiple files and create a ZIP
        files = [('file1.csv', data1, 'text/csv'), ('file2.csv', data2, 'text/csv')]
        zip_url = manager.upload_and_zip(files, 'batch_results')
    """
    
    def __init__(self, bucket_name: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize S3 Manager.
        
        Args:
            bucket_name: S3 bucket name (defaults to config)
            region: AWS region (defaults to config)
        """
        self.bucket_name = bucket_name or s3_config.bucket_name
        self.region = region or s3_config.aws_region
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=s3_config.aws_access_key_id,
                aws_secret_access_key=s3_config.aws_secret_access_key
            )
        except NoCredentialsError:
            raise RuntimeError(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "environment variables or configure AWS credentials file."
            )
    
    def _generate_s3_key(self, filename: str, folder: Optional[str] = None) -> str:
        """
        Generate a unique S3 key for the file.
        
        Args:
            filename: Original filename
            folder: Optional subfolder in S3
            
        Returns:
            S3 key path
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = filename.replace(' ', '_')
        
        folder_prefix = folder or s3_config.upload_folder
        return f"{folder_prefix}/{timestamp}_{unique_id}_{safe_filename}"
    
    def upload_file(
        self, 
        file_data: bytes, 
        filename: str, 
        content_type: str = 'application/octet-stream',
        folder: Optional[str] = None
    ) -> str:
        """
        Upload a single file to S3.
        
        Args:
            file_data: File content as bytes
            filename: Name for the file
            content_type: MIME type of the file
            folder: Optional subfolder in S3
            
        Returns:
            Presigned URL for downloading the file
        """
        s3_key = self._generate_s3_key(filename, folder)
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_data,
                ContentType=content_type,
                Metadata={
                    'original_filename': filename,
                    'upload_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Generate presigned URL for download
            url = self._generate_presigned_url(s3_key)
            return url
            
        except ClientError as e:
            raise RuntimeError(f"Failed to upload file to S3: {str(e)}")
    
    def upload_and_zip(
        self, 
        files: List[Tuple[str, bytes, str]], 
        zip_name: str = 'generated_files',
        folder: Optional[str] = None
    ) -> str:
        """
        Create a ZIP file from multiple files and upload to S3.
        
        Args:
            files: List of tuples (filename, file_data, content_type)
            zip_name: Base name for the ZIP file (without extension)
            folder: Optional subfolder in S3
            
        Returns:
            Presigned URL for downloading the ZIP file
        """
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, file_data, _ in files:
                zip_file.writestr(filename, file_data)
        
        zip_data = zip_buffer.getvalue()
        zip_filename = f"{zip_name}.zip"
        
        # Upload ZIP to S3
        return self.upload_file(
            file_data=zip_data,
            filename=zip_filename,
            content_type='application/zip',
            folder=folder
        )
    
    def _generate_presigned_url(self, s3_key: str, expiration: Optional[int] = None) -> str:
        """
        Generate a presigned URL for downloading a file from S3.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default from config)
            
        Returns:
            Presigned URL
        """
        expiration = expiration or s3_config.presigned_url_expiration
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
            
        except ClientError as e:
            raise RuntimeError(f"Failed to generate presigned URL: {str(e)}")
    
    def upload_images_and_zip(
        self, 
        images: List[Tuple[str, BinaryIO]], 
        zip_name: str = 'generated_images',
        folder: Optional[str] = None
    ) -> str:
        """
        Upload PIL Images to S3 as a ZIP file.
        
        Args:
            images: List of tuples (filename, PIL.Image)
            zip_name: Base name for the ZIP file
            folder: Optional subfolder in S3
            
        Returns:
            Presigned URL for downloading the ZIP file
        """
        from PIL import Image
        
        files = []
        for filename, img in images:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            if isinstance(img, Image.Image):
                img.save(img_byte_arr, format='PNG')
            else:
                # Assume it's already bytes
                img_byte_arr.write(img)
            
            files.append((filename, img_byte_arr.getvalue(), 'image/png'))
        
        return self.upload_and_zip(files, zip_name, folder)
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
            
        except ClientError as e:
            raise RuntimeError(f"Failed to delete file from S3: {str(e)}")
    
    def list_files(self, prefix: Optional[str] = None, max_keys: int = 100) -> List[dict]:
        """
        List files in the S3 bucket.
        
        Args:
            prefix: Filter by prefix/folder
            max_keys: Maximum number of files to return
            
        Returns:
            List of file metadata dictionaries
        """
        prefix = prefix or s3_config.upload_folder
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'filename': obj['Key'].split('/')[-1]
                })
            
            return files
            
        except ClientError as e:
            raise RuntimeError(f"Failed to list files from S3: {str(e)}")
