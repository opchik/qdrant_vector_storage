"""Common exceptions for both sync and async clients."""


class QdrantError(Exception):
    """Base exception for Qdrant client."""
    pass


class CollectionNotFoundError(QdrantError):
    """Raised when collection does not exist."""
    pass


class CollectionExistsError(QdrantError):
    """Raised when collection already exists."""
    pass


class FileProcessingError(QdrantError):
    """Raised when file processing fails."""
    pass


class UnsupportedFileTypeError(FileProcessingError):
    """Raised when file type is not supported."""
    pass


class EmbeddingError(QdrantError):
    """Raised when embedding generation fails."""
    pass


class ConnectionError(QdrantError):
    """Raised when connection to Qdrant fails."""
    pass


class ValidationError(QdrantError):
    """Raised when validation fails."""
    pass