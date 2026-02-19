"""
Qdrant File Uploader - Sync and Async client for Qdrant vector database with file upload support.
"""

from .sync_client.client import QdrantSyncClient
from .async_client.client import QdrantAsyncClient
from .common.models import (
    SearchResult,
    FileUploadResult,
    FileType,
    Distance,
    TextChunk,
    EmbeddingModel
)
from .common.exceptions import (
    QdrantError,
    CollectionNotFoundError,
    CollectionExistsError,
    FileProcessingError,
    UnsupportedFileTypeError,
    EmbeddingError,
    ConnectionError
)

__version__ = "0.1.0"

__all__ = [
    "QdrantSyncClient",
    "QdrantAsyncClient",
    
    "CollectionConfig",
    "Document",
    "SearchResult",
    "FileUploadResult",
    "FileType",
    "Distance",
    "TextChunk",
    "EmbeddingModel",
    
    "QdrantError",
    "CollectionNotFoundError",
    "CollectionExistsError",
    "FileProcessingError",
    "UnsupportedFileTypeError",
    "EmbeddingError",
    "ConnectionError",
]