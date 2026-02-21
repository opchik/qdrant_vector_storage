import asyncio
from pathlib import Path

from fastembed import TextEmbedding

from qdrant_vector_storage.async_client.client import QdrantAsyncClient
from qdrant_vector_storage.common.base import MarkdownProcessor
from qdrant_vector_storage.common.converters import chunks_to_points


# -------------------- Config --------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "НСИ Общие документы"  # лучше без кириллицы и пробелов для теста
MD_FILE = "README.md"

# Выберите модель fastembed (должна быть в списке supported_models)
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
EXPECTED_DIM = 1024

# Для E5-моделей нужно True, для jina v3 — False
ADD_PASSAGE_PREFIX = False


# -------------------- Helpers --------------------

def print_supported_models(limit: int = 10) -> None:
    print("Fastembed supported models (first N):")
    i = 0
    for m in TextEmbedding.list_supported_models():
        i += 1
        print(f"{i:3d}. {m['model']}")
        if i >= limit:
            break
    print()


# -------------------- Main --------------------

async def main() -> None:
    print_supported_models(limit=8)

    if not Path(MD_FILE).exists():
        raise FileNotFoundError(f"Markdown file not found: {MD_FILE}")

    # 1) Create embedder
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # 2) Create processor (embedder is injected)
    processor = MarkdownProcessor(
        embedder=embedder,
        expected_dim=EXPECTED_DIM,
        chunk_size=900,
        chunk_overlap=120,
        batch_size=64,
        passage_prefix="passage: ",
    )

    # 3) Connect to Qdrant
    client = QdrantAsyncClient(url=QDRANT_URL)

    # 4) Create collection if missing
    exists = await client.client.collection_exists(COLLECTION_NAME)
    if not exists:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vector_size=EXPECTED_DIM,
        )

    # 5) Build chunks -> points
    chunks = processor.build_chunks(
        MD_FILE,
        source_name=MD_FILE,
        add_passage_prefix=ADD_PASSAGE_PREFIX,
    )
    points = chunks_to_points(chunks)

    # 6) Upload points
    point_ids = await client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=100,
        wait=True,
    )

    print(f"Uploaded points: {len(point_ids)}")

    # 7) Collection info
    info = await client.get_collection_info(collection_name=COLLECTION_NAME)
    print(f"collection_info.name: {info.get('config', {}).get('params', {}).get('vectors', {}).get('size', 'n/a')}")
    print(f"collection_info: {info}")

    # 8) List collections
    collections = await client.list_collections()
    print(f"collections: {collections}")

    # 9) Count points
    total_points = await client.count_points(collection_name=COLLECTION_NAME)
    print(f"total_points: {total_points}")

    # 10) Healthcheck
    flag = await client.healthcheck()
    print(f"healthcheck: {flag}")

    # 11) Optional: quick search check
    query_text = "о чем этот документ"
    query_vec = next(embedder.embed([query_text])).tolist()

    results = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=5,
        with_payload=True,
    )
    print("\nTop results:")
    for r in results:
        print(f"- score={r.score:.4f} text_snippet={r.text[:120]!r}")


if __name__ == "__main__":
    asyncio.run(main())