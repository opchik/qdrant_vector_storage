"""qdrant_vector_storage package.

Sync and async clients for Qdrant with Markdown chunking utilities.
"""

from .sync_client.client import QdrantSyncClient
from .async_client.client import QdrantAsyncClient

from .common.base import MarkdownProcessor, FilterBuilder
from .common.models import (
    Point,
    SearchResult,
    FileUploadResult,
    FileType,
    Distance,
    TextChunk,
)
from .common.exceptions import (
    QdrantError,
    CollectionNotFoundError,
    CollectionExistsError,
    FileProcessingError,
    UnsupportedFileTypeError,
    EmbeddingError,
    ConnectionError,
)

__version__ = "0.1.1"

__all__ = [
    "QdrantSyncClient",
    "QdrantAsyncClient",
    "MarkdownProcessor",
    "FilterBuilder",
    "Point",
    "SearchResult",
    "FileUploadResult",
    "FileType",
    "Distance",
    "TextChunk",
    "QdrantError",
    "CollectionNotFoundError",
    "CollectionExistsError",
    "FileProcessingError",
    "UnsupportedFileTypeError",
    "EmbeddingError",
    "ConnectionError",
]
