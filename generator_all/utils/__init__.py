"""
Utility modules for the generator application.
"""

from .s3_storage import S3Manager
from .helpers import get_paragraphs

__all__ = ['S3Manager', 'get_paragraphs']
