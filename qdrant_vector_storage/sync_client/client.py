"""Synchronous Qdrant client implementation."""

import uuid
import logging
from typing import Optional, List, Dict, Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from ..common.base import BaseParser, FilterBuilder
from ..common.models import Point, SearchResult, FileUploadResult, FileType, Distance
from ..common.exceptions import (
    QdrantError, CollectionNotFoundError, CollectionExistsError,
    FileProcessingError, EmbeddingError, ConnectionError
)

logger = logging.getLogger(__name__)


class QdrantSyncClient:
    """
    Synchronous Qdrant client for vector database operations.
    
    Supports:
    - Collection management
    - File upload (MD, JSON, HTML, LaTeX)
    - Point deletion
    - Vector search
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        **kwargs
    ):
        """
        Initialize sync Qdrant client.
        
        Args:
            url: Full Qdrant URL (e.g., http://localhost:6333)
            api_key: API key for authentication
            timeout: Request timeout in seconds
            **kwargs: Additional arguments for QdrantClient
        """
        try:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
                **kwargs
            )
            logger.info("Qdrant connection started")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise ConnectionError(f"Connection failed: {e}")
    
    # ==================== Collection Management ====================
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        on_disk_payload: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Vector dimension
            distance: Distance metric
            on_disk_payload: Store payload on disk
            **kwargs: Additional collection parameters
            
        Returns:
            Collection info
            
        Raises:
            CollectionExistsError: If collection already exists
            QdrantError: If creation fails
        """
        try:
            exists = self.client.collection_exists(collection_name)
            if exists:
                raise CollectionExistsError(f"Collection '{collection_name}' already exists")
            if not isinstance(vector_size, int) or vector_size <= 0 or vector_size > 65535:
                raise ValueError(
                    f"vector_size must be positive integer ≤ 65535, got {vector_size}"
                )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance.value
                ),
                on_disk_payload=on_disk_payload,
                **kwargs
            )
            return self.get_collection_info(collection_name)
            
        except CollectionExistsError:
            raise
        except Exception as e:
            logger.error("Failed to create collection '%s': %s", collection_name, e, exc_info=True)
            raise QdrantError(f"Collection creation failed: {e}") from e
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(collection_name)
            return info.dict()
        except UnexpectedResponse as e:
            if "Not found" in str(e):
                raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
            raise QdrantError(f"Failed to get collection info: {e}")
        except Exception as e:
            raise QdrantError(f"Failed to get collection info: {e}")
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            raise QdrantError(f"Failed to list collections: {e}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    # ==================== Point Upload ====================
    
    def upload_points(
        self,
        collection_name: str,
        points: List[Point],
        batch_size: int = 100,
        wait: bool = True
    ) -> List[str]:
        """
        Upload points to collection.
        
        Args:
            collection_name: Collection name
            points: List of points
            batch_size: Upload batch size
            wait: Wait for completion
            
        Returns:
            List of point IDs
        """
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        if not points:
            return []
        for point in points:
            if point.vector is None:
                raise ValueError(f"Point {point.id} has no vector")
            if not point.text:
                raise ValueError(f"Point {point.id} has no text")
        try:
            cur_points = []
            point_ids = []
            for point in points:
                point_id = point.id or str(uuid.uuid4())
                point_ids.append(point_id)
                cur_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=point.vector,
                        payload={
                            "text": point.text,
                            "metadata": point.metadata
                        }
                    )
                )
            for i in range(0, len(cur_points), batch_size):
                batch = cur_points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=wait
                )
            return point_ids
        except Exception as e:
            logger.error(f"Failed to upload points: {e}")
            raise QdrantError(f"Point upload failed: {e}") from e
    
    def upload_file(
        self,
        collection_name: str,
        file_path: str,
        embedder: Callable[[str], List[float]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> FileUploadResult:
        """
        Upload file to collection.
        
        Supported formats: MD
        
        Args:
            collection_name: Collection name
            file_path: Path to file
            embedder: Function to generate embeddings
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap
            metadata: Additional metadata
            batch_size: Upload batch size
            
        Returns:
            Upload result
        """
        try:
            # Parse file
            chunks = BaseParser.parse_file(file_path, chunk_size, chunk_overlap)
            
            if not chunks:
                raise FileProcessingError("No text content extracted from file")
            
            # Create points with embeddings
            points = []
            file_metadata = {
                "source": file_path,
                "file_name": file_path.split('/')[-1],
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                **(metadata or {})
            }
            
            for chunk in chunks:
                # Generate embedding
                try:
                    vector = embedder(chunk.text)
                except Exception as e:
                    raise EmbeddingError(f"Failed to generate embedding: {e}")
                
                # Combine metadata
                chunk_metadata = {
                    **file_metadata,
                    **chunk.metadata,
                    "chunk_index": chunk.index
                }
                
                doc = Point(
                    text=chunk.text,
                    metadata=chunk_metadata,
                    vector=vector
                )
                points.append(doc)
            
            # Upload points
            doc_ids = self.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=batch_size
            )
            
            # Determine file type
            ext = file_path.split('.')[-1].lower()
            file_type = FileType.MD
            if ext == 'json':
                file_type = FileType.JSON
            elif ext in ['html', 'htm']:
                file_type = FileType.HTML
            elif ext in ['tex', 'latex']:
                file_type = FileType.LATEX
            
            result = FileUploadResult(
                file_name=file_path.split('/')[-1],
                file_type=file_type,
                chunks_uploaded=len(points),
                point_ids=doc_ids,
                collection_name=collection_name
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise
    
    # ==================== Point Deletion ====================
    
    def delete_points(
        self,
        collection_name: str,
        point_ids: Optional[List[str]] = None,
        filter_condition: Optional[Dict[str, Any]] = None,
        wait: bool = True
    ) -> int:
        """
        Delete points from collection.
        
        Args:
            collection_name: Collection name
            point_ids: List of point IDs to delete
            filter_condition: Filter for deletion
            wait: Wait for completion
            
        Returns:
            Number of deleted points
        """
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        try:
            if point_ids:
                result = self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    ),
                    wait=wait
                )
            elif filter_condition:
                qdrant_filter = FilterBuilder.build_filter(filter_condition)
                result = self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=qdrant_filter
                    ),
                    wait=wait
                )
            else:
                raise ValueError("Either point_ids or filter_condition required")
            if hasattr(result, 'status') and hasattr(result.status, 'deleted'):
                deleted_count = result.status.deleted
            elif hasattr(result, 'result') and isinstance(result.result, dict):
                deleted_count = result.result.get('deleted', 0)
            else:
                deleted_count = 0
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            raise QdrantError(f"Deletion failed: {e}") from e
    
    def delete_by_metadata(
        self,
        collection_name: str,
        metadata_key: str,
        metadata_value: Any,
        wait: bool = True
    ) -> int:
        """
        Delete points by metadata value.
        
        Args:
            collection_name: Collection name
            metadata_key: Metadata key
            metadata_value: Value to match
            wait: Wait for completion
            
        Returns:
            Number of deleted points
        """
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        
        if not metadata_key:
            raise ValueError("metadata_key cannot be empty")
        
        if metadata_value is None:
            raise ValueError("metadata_value cannot be None")
        
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        
        filter_condition = {
            metadata_key: metadata_value
        }
        try:
            return self.delete_points(
                collection_name=collection_name,
                filter_condition=filter_condition,
                wait=wait
            )
        except ValueError as e:
            logger.error("Invalid filter for '%s': %s", collection_name, e)
            raise ValueError(f"Invalid metadata filter: {e}") from e
        except Exception as e:
            logger.error("Failed to delete by metadata from '%s': %s", collection_name, e, exc_info=True)
            raise QdrantError(f"Failed to delete by metadata: {e}") from e
    
    # ==================== Search ====================
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_condition: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection name
            query_vector: Query vector
            limit: Max results
            score_threshold: Minimum score
            filter_condition: Filter
            with_payload: Include payload
            with_vectors: Include vectors
            
        Returns:
            List of search results
        """
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        
        if not query_vector:
            raise ValueError("query_vector cannot be empty")
        
        if limit <= 0:
            raise ValueError("limit must be positive")
        
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        
        try:
            search_filter = None
            if filter_condition:
                search_filter = FilterBuilder.build_filter(filter_condition)
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            results = []
            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    SearchResult(
                        id=str(hit.id),
                        score=hit.score,
                        text=payload.get("text", ""),
                        metadata=payload.get("metadata", {})
                    )
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise QdrantError(f"Search failed: {e}")
        
    # ==================== Utility Methods ====================
    
    def count_points(
        self,
        collection_name: str,
        filter_condition: Optional[Dict[str, Any]] = None,
        exact: bool = False
    ) -> int:
        """Count points in collection."""
        try:
            if not self.client.collection_exists(collection_name):
                raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
            
            search_filter = None
            if filter_condition:
                search_filter = FilterBuilder.build_filter(filter_condition)
            
            result = self.client.count(
                collection_name=collection_name,
                count_filter=search_filter,
                exact=exact
            )
            return result.count
        except Exception as e:
            logger.error("Failed to count points in '%s': %s", collection_name, e)
            raise QdrantError(f"Count operation failed: {e}") from e
    
    def healthcheck(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def close(self):
        """Close client connection."""
        self.client.close()
        logger.info("Qdrant connection closed")
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()