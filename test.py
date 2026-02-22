import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from fastembed import TextEmbedding

from qdrant_vector_storage.async_client.client import QdrantAsyncClient
from qdrant_vector_storage.sync_client.client import QdrantSyncClient
from qdrant_vector_storage.common.base import MarkdownProcessor, FilterBuilder
from qdrant_vector_storage.common.models import Distance, SearchResult

# -------------------- Logging Setup --------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------- Configuration --------------------

@dataclass
class Config:
    """Configuration for the test."""
    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "НСИ Общие докуметы"  # используем латиницу для надёжности
    
    # Embedding model
    embedding_model: str = "jinaai/jina-embeddings-v3"
    expected_dim: int = 1024
    
    # Markdown processing
    chunk_size: int = 900
    chunk_overlap: int = 120
    batch_size: int = 64
    add_passage_prefix: bool = False  # для jina v3 не нужно
    
    # Test file
    md_file: str = "README.md"
    
    # Test filters
    test_chunk_index: int = 19
    test_source: str = "README.md"


# -------------------- Test Base Class --------------------

class QdrantTestBase:
    """Base class for Qdrant client tests."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedder = None
        self.processor = None
        self._setup_embedder()
    
    def _setup_embedder(self) -> None:
        """Initialize embedder and processor."""
        logger.info(f"Initializing embedder: {self.config.embedding_model}")
        self.embedder = TextEmbedding(model_name=self.config.embedding_model)
        
        self.processor = MarkdownProcessor(
            embedder=self.embedder,
            expected_dim=self.config.expected_dim,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            batch_size=self.config.batch_size,
            passage_prefix="passage: ",
        )
    
    def _check_md_file(self) -> None:
        """Check if markdown file exists."""
        if not Path(self.config.md_file).exists():
            raise FileNotFoundError(f"Markdown file not found: {self.config.md_file}")
    
    def print_supported_models(self, limit: int = 10) -> None:
        """Print supported embedding models."""
        logger.info("Fastembed supported models (first %d):", limit)
        for i, model in enumerate(TextEmbedding.list_supported_models(), 1):
            if i > limit:
                break
            logger.info(f"  {i:3d}. {model['model']}")
    
    def print_collection_info(self, info: Dict[str, Any]) -> None:
        """Pretty print collection info."""
        vector_size = info.get('config', {}).get('params', {}).get('vectors', {}).get('size', 'n/a')
        points_count = info.get('points_count', 0)
        logger.info(f"Collection: {self.config.collection_name}")
        logger.info(f"  Vector size: {vector_size}")
        logger.info(f"  Points count: {points_count}")


# -------------------- Sync Tests --------------------

class QdrantSyncTester(QdrantTestBase):
    """Test suite for synchronous Qdrant client."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.client: Optional[QdrantSyncClient] = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def connect(self) -> None:
        """Connect to Qdrant."""
        logger.info(f"Connecting to Qdrant at {self.config.qdrant_url}")
        self.client = QdrantSyncClient(url=self.config.qdrant_url)
    
    def close(self) -> None:
        """Close connection."""
        if self.client:
            self.client.close()
            logger.info("Connection closed")
    
    def ensure_collection(self) -> bool:
        """Ensure collection exists, create if not."""
        exists = self.client.client.collection_exists(self.config.collection_name)
        if not exists:
            logger.info(f"Creating collection: {self.config.collection_name}")
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vector_size=self.config.expected_dim,
                distance=Distance.COSINE,
            )
            return True
        else:
            logger.info(f"Collection already exists: {self.config.collection_name}")
            return False
    
    def test_upload_markdown(self) -> List[str]:
        """Test markdown upload."""
        logger.info("Testing markdown upload...")
        self._check_md_file()
        
        result = self.client.upload_markdown(
            collection_name=self.config.collection_name,
            md_input=self.config.md_file,
            processor=self.processor,
            source_name=self.config.md_file,
            batch_size=100,
            wait=True,
            processor_kwargs={"add_passage_prefix": self.config.add_passage_prefix}
        )
        
        logger.info(f"Uploaded {result.chunks_uploaded} chunks")
        logger.info(f"Point IDs: {result.point_ids[:5]}...")
        return result.point_ids
    
    def test_count_points(self, filter_condition: Optional[Dict] = None) -> int:
        """Test counting points."""
        count = self.client.count_points(
            collection_name=self.config.collection_name,
            filter_condition=filter_condition
        )
        logger.info(f"Points count (filter={filter_condition}): {count}")
        return count
    
    def test_search(self, query_text: str, limit: int = 5) -> List[SearchResult]:
        """Test vector search."""
        logger.info(f"Searching for: '{query_text}'")
        
        query_vec = next(self.embedder.embed([query_text])).tolist()
        
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        
        logger.info(f"Found {len(results)} results")
        for i, r in enumerate(results, 1):
            logger.info(f"  {i}. score={r.score:.4f} | {r.text[:80]}...")
        
        return results
    
    def test_filter_formats(self) -> None:
        """Test different filter formats."""
        logger.info("Testing filter formats...")
        
        # Simple dict
        filter1 = FilterBuilder.build_filter({"chunk_index": self.config.test_chunk_index})
        logger.info(f"Simple dict: {filter1}")
        
        # List of conditions
        filter2 = FilterBuilder.build_filter([
            {"chunk_index": self.config.test_chunk_index},
            {"source": self.config.test_source}
        ])
        logger.info(f"List of conditions: {filter2}")
        
        # Full format with operators
        filter3 = FilterBuilder.build_filter({
            "must": [
                {"key": "chunk_index", "match": {"value": self.config.test_chunk_index}}
            ],
            "should": [
                {"key": "source", "match": {"value": self.config.test_source}}
            ]
        })
        logger.info(f"Full format: {filter3}")
    
    def test_delete_by_filter(self) -> int:
        """Test deleting points by filter."""
        logger.info(f"Testing delete by filter: chunk_index={self.config.test_chunk_index}")
        
        # Count before deletion
        before = self.client.count_points(
            collection_name=self.config.collection_name,
            filter_condition={"chunk_index": self.config.test_chunk_index}
        )
        logger.info(f"Points to delete: {before}")
        
        if before == 0:
            logger.warning("No points to delete")
            return 0
        
        deleted = self.client.delete_points(
            collection_name=self.config.collection_name,
            filter_condition={"chunk_index": self.config.test_chunk_index},
            wait=True
        )
        
        # Count after deletion
        after = self.client.count_points(
            collection_name=self.config.collection_name,
        )
        
        logger.info(f"Deleted: {deleted}, Remaining: {after}")
        return deleted
    
    def test_delete_by_metadata(self) -> int:
        """Test deleting by metadata convenience method."""
        logger.info(f"Testing delete_by_metadata: source={self.config.test_source}")
        
        deleted = self.client.delete_by_metadata(
            collection_name=self.config.collection_name,
            metadata_key="source",
            metadata_value=self.config.test_source,
            wait=True
        )
        
        logger.info(f"Deleted by metadata: {deleted}")
        return deleted
    
    def test_delete_by_ids(self, point_ids: List[str]) -> int:
        """Test deleting by point IDs."""
        if not point_ids:
            logger.warning("No point IDs provided")
            return 0
        
        logger.info(f"Testing delete by IDs: {point_ids[:3]}...")
        
        deleted = self.client.delete_points(
            collection_name=self.config.collection_name,
            point_ids=point_ids[:3],
            wait=True
        )
        
        logger.info(f"Deleted by IDs: {deleted}")
        return deleted
    
    def run_all_tests(self) -> None:
        """Run all tests in sequence."""
        logger.info("=" * 60)
        logger.info("Starting synchronous client tests")
        logger.info("=" * 60)
        
        try:
            # 1. Healthcheck
            assert self.client.healthcheck(), "Healthcheck failed"
            logger.info("✓ Healthcheck passed")
            
            # 2. Ensure collection
            created = self.ensure_collection()
            if created:
                logger.info("✓ Collection created")
            else:
                logger.info("✓ Collection exists")
            
            # 3. Collection info
            info = self.client.get_collection_info(self.config.collection_name)
            self.print_collection_info(info)
            
            # 4. List collections
            collections = self.client.list_collections()
            logger.info(f"Collections: {collections}")
            
            # 5. Initial count
            initial_count = self.test_count_points()
            
            # 6. Upload if collection is empty
            if initial_count == 0:
                point_ids = self.test_upload_markdown()
                logger.info(f"✓ Uploaded {len(point_ids)} points")
            else:
                logger.info(f"Collection already has {initial_count} points, skipping upload")
                # Get some point IDs for testing
                # (в реальном коде можно получить через scroll)
                point_ids = []
            
            # 7. Count after upload
            after_upload = self.test_count_points()
            logger.info(f"✓ Count after upload: {after_upload}")
            
            # 8. Test search
            self.test_search("о чем этот документ", limit=5)
            
            # 9. Test filter formats
            self.test_filter_formats()
            
            # 10. Test count with filter
            count_filtered = self.test_count_points({"chunk_index": self.config.test_chunk_index})
            
            # 11. Test delete by filter (if points exist)
            if count_filtered > 0:
                deleted = self.test_delete_by_filter()
                logger.info(f"✓ Deleted {deleted} points by filter")
            else:
                logger.info(f"No points with chunk_index={self.config.test_chunk_index} to delete")
            
            # 12. Final count
            final_count = self.test_count_points()
            logger.info(f"Final points count: {final_count}")
            
            logger.info("=" * 60)
            logger.info("All synchronous tests completed successfully")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            raise


# -------------------- Async Tests --------------------

class QdrantAsyncTester(QdrantTestBase):
    """Test suite for asynchronous Qdrant client."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.client: Optional[QdrantAsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Connect to Qdrant."""
        logger.info(f"Connecting to Qdrant at {self.config.qdrant_url}")
        self.client = QdrantAsyncClient(url=self.config.qdrant_url)
    
    async def close(self) -> None:
        """Close connection."""
        if self.client:
            await self.client.close()
            logger.info("Connection closed")
    
    async def ensure_collection(self) -> bool:
        """Ensure collection exists, create if not."""
        exists = await self.client.client.collection_exists(self.config.collection_name)
        if not exists:
            logger.info(f"Creating collection: {self.config.collection_name}")
            await self.client.create_collection(
                collection_name=self.config.collection_name,
                vector_size=self.config.expected_dim,
                distance=Distance.COSINE,
            )
            return True
        else:
            logger.info(f"Collection already exists: {self.config.collection_name}")
            return False
    
    async def test_upload_markdown(self) -> List[str]:
        """Test markdown upload."""
        logger.info("Testing markdown upload...")
        self._check_md_file()
        
        result = await self.client.upload_markdown(
            collection_name=self.config.collection_name,
            md_input=self.config.md_file,
            processor=self.processor,
            source_name=self.config.md_file,
            batch_size=100,
            wait=True,
            processor_kwargs={"add_passage_prefix": self.config.add_passage_prefix}
        )
        
        logger.info(f"Uploaded {result.chunks_uploaded} chunks")
        logger.info(f"Point IDs: {result.point_ids[:5]}...")
        return result.point_ids
    
    async def test_count_points(self, filter_condition: Optional[Dict] = None) -> int:
        """Test counting points."""
        count = await self.client.count_points(
            collection_name=self.config.collection_name,
            filter_condition=filter_condition
        )
        logger.info(f"Points count (filter={filter_condition}): {count}")
        return count
    
    async def test_search(self, query_text: str, limit: int = 5) -> List[SearchResult]:
        """Test vector search."""
        logger.info(f"Searching for: '{query_text}'")
        
        query_vec = next(self.embedder.embed([query_text])).tolist()
        
        results = await self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        
        logger.info(f"Found {len(results)} results")
        for i, r in enumerate(results, 1):
            logger.info(f"  {i}. score={r.score:.4f} | {r.text[:80]}...")
        
        return results
    
    async def test_delete_by_filter(self) -> int:
        """Test deleting points by filter."""
        logger.info(f"Testing delete by filter: chunk_index={self.config.test_chunk_index}")
        
        before = await self.client.count_points(
            collection_name=self.config.collection_name,
            filter_condition={"chunk_index": self.config.test_chunk_index}
        )
        logger.info(f"Points to delete: {before}")
        
        if before == 0:
            logger.warning("No points to delete")
            return 0
        
        deleted = await self.client.delete_points(
            collection_name=self.config.collection_name,
            filter_condition={"chunk_index": self.config.test_chunk_index},
            wait=True
        )
        
        after = await self.client.count_points(
            collection_name=self.config.collection_name,
            filter_condition={"chunk_index": self.config.test_chunk_index}
        )
        
        logger.info(f"Deleted: {deleted}, Remaining: {after}")
        return deleted
    
    async def run_all_tests(self) -> None:
        """Run all tests in sequence."""
        logger.info("=" * 60)
        logger.info("Starting asynchronous client tests")
        logger.info("=" * 60)
        
        try:
            # 1. Healthcheck
            assert await self.client.healthcheck(), "Healthcheck failed"
            logger.info("✓ Healthcheck passed")
            
            # 2. Ensure collection
            created = await self.ensure_collection()
            if created:
                logger.info("✓ Collection created")
            else:
                logger.info("✓ Collection exists")
            
            # 3. Collection info
            info = await self.client.get_collection_info(self.config.collection_name)
            self.print_collection_info(info)
            
            # 4. List collections
            collections = await self.client.list_collections()
            logger.info(f"Collections: {collections}")
            
            # 5. Initial count
            initial_count = await self.test_count_points()
            
            # 6. Upload if collection is empty
            if initial_count == 0:
                point_ids = await self.test_upload_markdown()
                logger.info(f"✓ Uploaded {len(point_ids)} points")
            
            # 7. Count after upload
            after_upload = await self.test_count_points()
            logger.info(f"✓ Count after upload: {after_upload}")
            
            # 8. Test search
            await self.test_search("о чем этот документ", limit=5)
            
            # 9. Test count with filter
            count_filtered = await self.test_count_points(
                {"chunk_index": self.config.test_chunk_index}
            )
            
            # 10. Test delete by filter (if points exist)
            if count_filtered > 0:
                deleted = await self.test_delete_by_filter()
                logger.info(f"✓ Deleted {deleted} points by filter")
            else:
                logger.info(f"No points with chunk_index={self.config.test_chunk_index} to delete")
            
            # 11. Final count
            final_count = await self.test_count_points()
            logger.info(f"Final points count: {final_count}")
            
            logger.info("=" * 60)
            logger.info("All asynchronous tests completed successfully")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            raise


# -------------------- Main Execution --------------------

def main():
    """Main entry point."""
    config = Config()
    
    # Choose which client to test
    test_sync = False
    test_async = True
    
    if test_sync:
        with QdrantSyncTester(config) as tester:
            tester.run_all_tests()
    
    if test_async:
        async def run_async():
            async with QdrantAsyncTester(config) as tester:
                await tester.run_all_tests()
        
        asyncio.run(run_async())


if __name__ == "__main__":
    main()