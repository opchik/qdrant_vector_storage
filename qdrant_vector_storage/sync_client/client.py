"""Synchronous Qdrant client implementation."""

import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from ..common.base import MarkdownProcessor, FilterBuilder
from ..common.converters import chunks_to_points
from ..common.models import Point, SearchResult, FileUploadResult, FileType, Distance
from ..common.exceptions import (
    QdrantError,
    CollectionNotFoundError,
    CollectionExistsError,
    FileProcessingError,
    EmbeddingError,
    ConnectionError,
)

logger = logging.getLogger(__name__)


class QdrantSyncClient:
    """Synchronous Qdrant client for vector database operations.

    Supports:
    - Collection management
    - Markdown upload (text / base64 / path) via MarkdownProcessor
    - Point deletion
    - Vector search
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        **kwargs: Any,
    ) -> None:
        """Initialize sync Qdrant client.

        Args:
            url: Full Qdrant URL (e.g., http://localhost:6333)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            **kwargs: Additional arguments for QdrantClient
        """
        try:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
                **kwargs,
            )
            logger.info("Qdrant connection started")
        except Exception as e:
            logger.error("Failed to initialize Qdrant client: %s", e, exc_info=True)
            raise ConnectionError(f"Connection failed: {e}") from e

    # ==================== Collection Management ====================

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        on_disk_payload: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new collection."""
        try:
            exists = self.client.collection_exists(collection_name)
            if exists:
                raise CollectionExistsError(f"Collection '{collection_name}' already exists")
            if not isinstance(vector_size, int) or vector_size <= 0 or vector_size > 65535:
                raise ValueError(f"vector_size must be positive integer ≤ 65535, got {vector_size}")

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance.value),
                on_disk_payload=on_disk_payload,
                **kwargs,
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
            raise QdrantError(f"Failed to get collection info: {e}") from e
        except Exception as e:
            raise QdrantError(f"Failed to get collection info: {e}") from e

    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            raise QdrantError(f"Failed to list collections: {e}") from e

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection '%s': %s", collection_name, e, exc_info=True)
            return False

    # ==================== Point Upload ====================

    def upload_points(
        self,
        collection_name: str,
        points: List[Point],
        batch_size: int = 100,
        wait: bool = True,
    ) -> List[str]:
        """Upload points to collection."""
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        if not points:
            return []

        for p in points:
            if p.vector is None:
                raise ValueError(f"Point {p.id} has no vector")
            if not p.text:
                raise ValueError(f"Point {p.id} has no text")

        try:
            cur_points: List[models.PointStruct] = []
            point_ids: List[str] = []
            for p in points:
                point_id = p.id or str(uuid.uuid4())
                point_ids.append(point_id)
                cur_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=p.vector,
                        payload={"text": p.text, "metadata": p.metadata or {}},
                    )
                )

            for i in range(0, len(cur_points), batch_size):
                batch = cur_points[i : i + batch_size]
                self.client.upsert(collection_name=collection_name, points=batch, wait=wait)
            return point_ids
        except Exception as e:
            logger.error("Failed to upload points: %s", e, exc_info=True)
            raise QdrantError(f"Point upload failed: {e}") from e

    def upload_markdown(
        self,
        collection_name: str,
        md_input: Union[str, "os.PathLike[str]"],
        processor: MarkdownProcessor,
        *,
        source_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        wait: bool = True,
        processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> FileUploadResult:
        """Upload Markdown (string/base64/path) to Qdrant using MarkdownProcessor.

        Args:
            collection_name: Target Qdrant collection.
            md_input: Markdown text, base64(Markdown), or path to .md.
            processor: MarkdownProcessor instance configured with an embedder.
            source_name: Optional logical source name stored in metadata.
            metadata: Optional metadata merged into each chunk metadata.
            batch_size: Upsert batch size.
            wait: Wait for upsert completion.
            processor_kwargs: Extra kwargs passed to processor.build_chunks (e.g. add_passage_prefix).

        Returns:
            FileUploadResult
        """
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        if processor is None:
            raise ValueError("processor cannot be None")

        try:
            pk = processor_kwargs or {}
            try:
                chunks = processor.build_chunks(md_input, source_name=source_name, **pk)
            except AttributeError as e:
                raise FileProcessingError(
                    "MarkdownProcessor должен иметь метод build_chunks(source, source_name=..., **kwargs)"
                ) from e
            except Exception as e:
                raise FileProcessingError(f"Failed to process markdown: {e}") from e

            if not chunks:
                raise FileProcessingError("No text content extracted from markdown input")

            base_meta: Dict[str, Any] = dict(metadata or {})
            if source_name:
                base_meta.setdefault("source", source_name)
            base_meta.setdefault("created_at", datetime.now().isoformat())

            try:
                points = chunks_to_points(chunks, base_metadata=base_meta)
            except Exception as e:
                raise EmbeddingError(f"Failed to build points from chunks: {e}") from e

            point_ids = self.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=batch_size,
                wait=wait,
            )

            file_name = source_name or f"markdown_input_{str(uuid.uuid4())}"
            return FileUploadResult(
                file_name=file_name,
                file_type=FileType.MD,
                chunks_uploaded=len(points),
                point_ids=point_ids,
                collection_name=collection_name,
            )
        except (CollectionNotFoundError, FileProcessingError, EmbeddingError):
            raise
        except Exception as e:
            logger.error("Failed to upload markdown: %s", e, exc_info=True)
            raise QdrantError(f"Markdown upload failed: {e}") from e

    # ==================== Point Deletion ====================

    def delete_points(
        self,
        collection_name: str,
        point_ids: Optional[List[str]] = None,
        filter_condition: Optional[Dict[str, Any]] = None,
        wait: bool = True,
    ) -> int:
        """Delete points from collection by ids or by filter."""
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        try:
            init_count = self.count_points(collection_name=collection_name)
            if point_ids:
                result = self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                    wait=wait,
                )
            elif filter_condition:
                qdrant_filter = FilterBuilder.build_filter(filter_condition)
                if qdrant_filter is not None:
                    result = self.client.delete(
                        collection_name=collection_name,
                        points_selector=models.FilterSelector(filter=qdrant_filter),
                        wait=wait,
                    )
            else:
                raise ValueError("Either point_ids or filter_condition required")
            if hasattr(result, 'status') and result.status.value == 'completed':
                return init_count - self.count_points(collection_name=collection_name)
            return 0
        except Exception as e:
            logger.error("Failed to delete points: %s", e, exc_info=True)
            raise QdrantError(f"Deletion failed: {e}") from e

    # ==================== Search ====================

    def search(
        self,
        collection_name: str,
        query_vector: Optional[List[float]] = None,
        query_point_id: Optional[Union[str, int]] = None,
        filter_condition: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        search_mode: Literal["vector", "id", "filter", "hybrid"] = "vector",
    ) -> List[SearchResult]:
        """
        Универсальный метод поиска с поддержкой различных типов запросов
        
        Args:
            collection_name: Имя коллекции
            query_vector: Вектор запроса (для векторного или гибридного поиска)
            query_point_id: ID точки для поиска похожих (рекомендации)
            filter_only: Поиск только по фильтру (без вектора)
            limit: Максимальное количество результатов
            score_threshold: Порог релевантности
            filter_condition: Условия фильтрации
            with_payload: Возвращать ли payload
            with_vectors: Возвращать ли векторы
            search_mode: Режим поиска:
                - "vector": только по вектору
                - "id": поиск похожих на точку по ID
                - "filter": только по фильтру (без вектора)
                - "hybrid": комбинированный поиск (вектор + фильтр)
            hybrid_weight: Вес векторного поиска в гибридном режиме (0-1)
        
        Returns:
            Список результатов поиска
        """
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        if search_mode == "vector" and query_vector is None:
            raise ValueError("query_vector is required for vector search")
        if search_mode == "id" and query_point_id is None:
            raise ValueError("query_point_id is required for id-based search")
        if search_mode == "filter" and filter_condition is None:
            raise ValueError("filter_only is required for filter-only search")
        if search_mode == "hybrid" and query_vector is None:
            raise ValueError("query_vector is required for hybrid search")
        try:
            search_filter = FilterBuilder.build_filter(filter_condition) if filter_condition else None
            if search_mode == "vector":
                hits = self._vector_search(
                    collection_name=collection_name, 
                    query_vector=query_vector, 
                    limit=limit, 
                    score_threshold=score_threshold, 
                    search_filter=search_filter,
                    with_payload=with_payload, 
                )
            elif search_mode == "id":
                hits = self._id_based_search(
                    collection_name=collection_name,
                    point_id=query_point_id, 
                    limit=limit, 
                    score_threshold=score_threshold,
                    search_filter=search_filter, 
                    with_payload=with_payload
                )
            elif search_mode == "filter":
                hits = self._filter_only_search(
                    collection_name=collection_name, 
                    filter_only=filter_condition, 
                    limit=limit, 
                    with_payload=with_payload
                )
            elif search_mode == "hybrid":
                hits = self._hybrid_search(
                    collection_name=collection_name, 
                    query_vector=query_vector, 
                    filter_condition=filter_condition, 
                    limit=limit,
                    score_threshold=score_threshold, 
                    with_payload=with_payload
                )
            else:
                raise ValueError(f"Unsupported search mode: {search_mode}")
            return self._process_results(hits)
        except Exception as e:
            logger.error("Search failed: %s", e, exc_info=True)
            raise QdrantError(f"Search failed: {e}") from e

    def _vector_search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int,
        score_threshold: Optional[float], 
        search_filter: Optional[models.Filter],
        with_payload: bool, 
    ):
        """Векторный поиск"""
        hits = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=search_filter,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )
        return hits.points

    def _id_based_search(
        self, 
        collection_name: str, 
        point_id: Union[str, int], 
        limit: int,
        score_threshold: Optional[float], 
        search_filter: Optional[models.Filter],
        with_payload: bool, 
    ):
        """Поиск похожих на точку по ID (рекомендации)"""
        point = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_vectors=True
        )
        if not point:
            return []
        query_vector = point[0].vector
        hits = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=search_filter,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )
        return hits.points

    def _filter_only_search(
        self, 
        collection_name: str, 
        filter_only: Dict[str, Any], 
        limit: int,
        with_payload: bool, 
    ):
        """Поиск только по фильтру (скроллинг)"""
        scroll_filter = FilterBuilder.build_filter(filter_only)
        points, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=with_payload,
        )
        return points

    def _hybrid_search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        filter_condition: Optional[Dict[str, Any]], 
        limit: int,
        score_threshold: Optional[float], 
        with_payload: bool, 
    ):
        """Гибридный поиск с использованием prefetch и fusion"""
        search_filter = FilterBuilder.build_filter(filter_condition) if filter_condition else None
        hits = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit * 2,
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )
        return hits.points

    def _process_results(self, points) -> List[SearchResult]:
        """Обработка результатов поиска"""
        results: List[SearchResult] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    id=str(point.id),
                    score=getattr(point, 'score', 1.0),
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", {})
                )
            )
        return results

    # ==================== Utility Methods ====================

    def count_points(
        self,
        collection_name: str,
        filter_condition: Optional[Dict[str, Any]] = None,
        exact: bool = False,
    ) -> int:
        """Count points in collection."""
        if not self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
        try:
            search_filter = FilterBuilder.build_filter(filter_condition) if filter_condition else None
            result = self.client.count(collection_name=collection_name, count_filter=search_filter, exact=exact)
            return int(result.count)
        except Exception as e:
            logger.error("Failed to count points in '%s': %s", collection_name, e, exc_info=True)
            raise QdrantError(f"Count operation failed: {e}") from e

    def healthcheck(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close client connection."""
        try:
            self.client.close()
        finally:
            logger.info("Qdrant connection closed")

    def __enter__(self) -> "QdrantSyncClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
