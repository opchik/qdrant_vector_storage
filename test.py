
import asyncio

from qdrant_vector_storage.async_client.client import QdrantAsyncClient
from qdrant_vector_storage.common.models import EmbeddingModel
from qdrant_vector_storage.common.base import MarkdownProcessor

collection_name = "НСИ Общие документы"
file_name = "README.md"


from fastembed import TextEmbedding
i = 0
for m in TextEmbedding.list_supported_models():
    i += 1
    print(f"{i:3d}. {m['model']}")
    print(m['description'])
    print()


processor = MarkdownProcessor(
    embedding_model=EmbeddingModel.INTFLOAT_MULTILINGUAL_E5_BASE,
    chunk_size=900,
    chunk_overlap=120,
    batch_size=64,
)


async def main():
    vector_type = EmbeddingModel.INTFLOAT_MULTILINGUAL_E5_SMALL
    client = QdrantAsyncClient(url='http://localhost:6333')
    # await client.create_collection(
    #     collection_name=collection_name,
    #     vector_size=vector_type.dimension,
    # )

    res3 = await client.upload_markdown(
        collection_name=collection_name,
        md_input=file_name,
        processor=processor,
        source_name=file_name,
    )
    print()
    print(f"res3: {res3}")
    print()

    info = await client.get_collection_info(collection_name=collection_name)
    print(f"collection_info: {info}")

    collections = await client.list_collections()
    print(f"collections: {collections}")



    total_points = await client.count_points(collection_name=collection_name)
    print(f"total_points: {total_points}")

    flag = await client.healthcheck()
    print(f"healthcheck: {flag}")

    # del_coll_flag = await client.delete_collection(collection_name=collection_name)
    # print(f"del_collection_flag: {del_coll_flag}")

if __name__=="__main__":
    # asyncio.run(main())
    pass

        
