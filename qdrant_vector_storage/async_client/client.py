"""
Asynchronous Qdrant client implementation.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from ..common.base import MarkdownProcessor, FilterBuilder
from ..common.models import Point, SearchResult, FileUploadResult, FileType, Distance
from ..common.exceptions import (
    QdrantError, CollectionNotFoundError, CollectionExistsError,
    FileProcessingError, EmbeddingError, ConnectionError
)

logger = logging.getLogger(__name__)


class QdrantAsyncClient:
    """
    Asynchronous Qdrant client for vector database operations.

    Supports:
    - Collection management
    - MD upload (text / base64 / path)
    - Point deletion
    - Vector search
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        **kwargs
    ):
        try:
            self.client = AsyncQdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
                **kwargs
            )
            logger.info("Async Qdrant connection started")
        except Exception as e:
            logger.error("Failed to initialize async Qdrant client: %s", e, exc_info=True)
            raise ConnectionError(f"Connection failed: {e}") from e

    # ==================== Collection Management ====================

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        on_disk_payload: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        try:
            exists = await self.client.collection_exists(collection_name)
            if exists:
                raise CollectionExistsError(f"Collection '{collection_name}' already exists")

            if not isinstance(vector_size, int) or vector_size <= 0 or vector_size > 65535:
                raise ValueError(f"vector_size must be positive integer ≤ 65535, got {vector_size}")

            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance.value
                ),
                on_disk_payload=on_disk_payload,
                **kwargs
            )
            return await self.get_collection_info(collection_name)

        except CollectionExistsError:
            raise
        except Exception as e:
            logger.error("Failed to create collection '%s': %s", collection_name, e, exc_info=True)
            raise QdrantError(f"Collection creation failed: {e}") from e

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            info = await self.client.get_collection(collection_name)
            return info.dict()
        except UnexpectedResponse as e:
            if "Not found" in str(e):
                raise CollectionNotFoundError(f"Collection '{collection_name}' not found") from e
            raise QdrantError(f"Failed to get collection info: {e}") from e
        except Exception as e:
            raise QdrantError(f"Failed to get collection info: {e}") from e

    async def list_collections(self) -> List[str]:
        try:
            collections = await self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            raise QdrantError(f"Failed to list collections: {e}") from e

    async def delete_collection(self, collection_name: str) -> bool:
        try:
            await self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection: %s", e, exc_info=True)
            return False

    # ==================== Point Upload ====================

    async def upload_points(
        self,
        collection_name: str,
        points: List[Point],
        batch_size: int = 100,
        wait: bool = True
    ) -> List[str]:
        if not collection_name:
            raise ValueError("collection_name cannot be empty")

        if not await self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

        if not points:
            return []

        for point in points:
            if point.vector is None:
                raise ValueError(f"Point {getattr(point, 'id', None)} has no vector")
            if not point.text:
                raise ValueError(f"Point {getattr(point, 'id', None)} has no text")

        try:
            cur_points: List[models.PointStruct] = []
            point_ids: List[str] = []

            for point in points:
                point_id = getattr(point, "id", None) or str(uuid.uuid4())
                point_ids.append(point_id)
                cur_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=point.vector,
                        payload={
                            "text": point.text,
                            "metadata": point.metadata or {},
                        }
                    )
                )

            for i in range(0, len(cur_points), batch_size):
                batch = cur_points[i:i + batch_size]
                await self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=wait
                )

            return point_ids
        except Exception as e:
            logger.error("Failed to upload points: %s", e, exc_info=True)
            raise QdrantError(f"Point upload failed: {e}") from e

    # ==================== Markdown Upload (text / base64 / path) ====================

    async def upload_markdown(
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
        """
        Загрузка Markdown в Qdrant, где md_input может быть:
          - строкой Markdown
          - строкой base64(Markdown)
          - путём к .md файлу

        processor: ваш MarkdownProcessor, который режет текст и считает эмбеддинги (fastembed)
                   и возвращает список чанков с vector.

        processor_kwargs: дополнительные параметры для processor (например, chunk_size, overlap и т.п.)
        """
        if not collection_name:
            raise ValueError("collection_name cannot be empty")

        if not await self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

        if processor is None:
            raise ValueError("processor cannot be None")

        try:
            # 1) Обработка markdown -> чанки с векторами
            try:
                pk = processor_kwargs or {}
                # Ожидаем API как в предыдущем решении:
                # processor.build_chunks(...) -> List[TextChunk] (text/index/metadata/vector)
                chunks = processor.build_chunks(
                    md_input,
                    source_name=source_name,
                    **pk,
                )
            except AttributeError as e:
                raise FileProcessingError(
                    "MarkdownProcessor должен иметь метод build_chunks(source, source_name=..., **kwargs)"
                ) from e
            except Exception as e:
                raise FileProcessingError(f"Failed to process markdown: {e}") from e

            if not chunks:
                raise FileProcessingError("No text content extracted from markdown input")

            # 2) Превращаем чанки -> points
            base_meta: Dict[str, Any] = dict(metadata or {})
            if source_name:
                base_meta.setdefault("source", source_name)

            points: List[Point] = []
            for ch in chunks:
                if getattr(ch, "vector", None) is None:
                    raise EmbeddingError(f"Chunk {getattr(ch, 'index', None)} has no vector")
                if not getattr(ch, "text", ""):
                    continue

                chunk_meta = {}
                chunk_meta.update(base_meta)
                # metadata из процессора (например, embedding_model, chunk_size...)
                chunk_meta.update(getattr(ch, "metadata", {}) or {})
                # индекс чанка — отдельно, чтобы удобно фильтровать
                chunk_meta["chunk_index"] = int(getattr(ch, "index", 0))
                chunk_meta["created_at"] = datetime.now()
                points.append(
                    Point(
                        id=str(uuid.uuid4()),
                        text=ch.text,
                        metadata=chunk_meta,
                        vector=ch.vector,
                    )
                )
            point_ids = await self.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=batch_size,
                wait=wait,
            )
            file_name = source_name or f"markdown_input_{str(uuid.uuid4())}"
            result = FileUploadResult(
                file_name=file_name,
                file_type=FileType.MD,
                chunks_uploaded=len(points),
                point_ids=point_ids,
                collection_name=collection_name,
            )
            return result
        except (CollectionNotFoundError, FileProcessingError, EmbeddingError):
            raise
        except Exception as e:
            logger.error("Failed to upload markdown: %s", e, exc_info=True)
            raise QdrantError(f"Markdown upload failed: {e}") from e

    # ==================== Point Deletion ====================

    async def delete_points(
        self,
        collection_name: str,
        point_ids: Optional[List[str]] = None,
        filter_condition: Optional[Dict[str, Any]] = None,
        wait: bool = True
    ) -> int:
        if not await self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

        try:
            if point_ids:
                result = await self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                    wait=wait
                )
            elif filter_condition:
                qdrant_filter = FilterBuilder.build_filter(filter_condition)
                result = await self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=qdrant_filter),
                    wait=wait
                )
            else:
                raise ValueError("Either point_ids or filter_condition required")

            if hasattr(result, "status") and hasattr(result.status, "deleted"):
                return result.status.deleted
            if hasattr(result, "result") and isinstance(result.result, dict):
                return int(result.result.get("deleted", 0))
            return 0

        except Exception as e:
            logger.error("Failed to delete points: %s", e, exc_info=True)
            raise QdrantError(f"Deletion failed: {e}") from e

    async def delete_by_metadata(
        self,
        collection_name: str,
        metadata_key: str,
        metadata_value: Any,
        wait: bool = True
    ) -> int:
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        if not metadata_key:
            raise ValueError("metadata_key cannot be empty")
        if metadata_value is None:
            raise ValueError("metadata_value cannot be None")
        if not await self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

        filter_condition = {metadata_key: metadata_value}

        try:
            return await self.delete_points(
                collection_name=collection_name,
                filter_condition=filter_condition,
                wait=wait
            )
        except ValueError as e:
            logger.error("Invalid filter for '%s': %s", collection_name, e, exc_info=True)
            raise ValueError(f"Invalid metadata filter: {e}") from e
        except Exception as e:
            logger.error("Failed to delete by metadata from '%s': %s", collection_name, e, exc_info=True)
            raise QdrantError(f"Failed to delete by metadata: {e}") from e

    # ==================== Search ====================

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_condition: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[SearchResult]:
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        if not query_vector:
            raise ValueError("query_vector cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not await self.client.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

        try:
            search_filter = FilterBuilder.build_filter(filter_condition) if filter_condition else None
            hits = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            results: List[SearchResult] = []
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
            logger.error("Search failed: %s", e, exc_info=True)
            raise QdrantError(f"Search failed: {e}") from e

    # ==================== Utility Methods ====================

    async def count_points(
        self,
        collection_name: str,
        filter_condition: Optional[Dict[str, Any]] = None,
        exact: bool = False
    ) -> int:
        try:
            if not await self.client.collection_exists(collection_name):
                raise CollectionNotFoundError(f"Collection '{collection_name}' not found")

            search_filter = FilterBuilder.build_filter(filter_condition) if filter_condition else None
            result = await self.client.count(
                collection_name=collection_name,
                count_filter=search_filter,
                exact=exact
            )
            return result.count
        except Exception as e:
            logger.error("Failed to count points in '%s': %s", collection_name, e, exc_info=True)
            raise QdrantError(f"Count operation failed: {e}") from e

    async def healthcheck(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self):
        await self.client.close()
        logger.info("Async Qdrant connection closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()