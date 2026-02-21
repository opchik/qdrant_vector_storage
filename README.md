# qdrant_vector_storage

Библиотека предоставляет **синхронный и асинхронный** клиент для Qdrant, а также готовый пайплайн:
**Markdown → чанки → эмбеддинги → загрузка в Qdrant**.

Ключевая идея: библиотека **не навязывает** конкретную модель эмбеддингов.  
Вы передаёте объект `embedder` (например, `fastembed.TextEmbedding`), а `MarkdownProcessor` использует его для расчёта векторов.

---

## Установка

Базовая установка (клиенты + модели данных):

```bash
pip install qdrant_vector_storage
```

С fastembed (рекомендуется для локальных эмбеддингов):

```bash
pip install "qdrant_vector_storage[fastembed]"
```

Опционально fastembed-gpu:

```bash
pip install "qdrant_vector_storage[fastembed-gpu]"
```

---

## Быстрый старт

### 1) Создайте embedder (пример: fastembed)

```python
from fastembed import TextEmbedding

embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v3")
```

### 2) Создайте MarkdownProcessor

```python
from qdrant_vector_storage import MarkdownProcessor

processor = MarkdownProcessor(
    embedder=embedder,
    expected_dim=1024,
    chunk_size=2000,
    chunk_overlap=200,
)
```

### 3) Асинхронная загрузка Markdown

```python
import asyncio
from qdrant_vector_storage import QdrantAsyncClient, Distance

async def main():
    async with QdrantAsyncClient(url="http://localhost:6333") as client:
        await client.create_collection(
            collection_name="docs",
            vector_size=1024,
            distance=Distance.COSINE,
        )

        result = await client.upload_markdown(
            collection_name="docs",
            md_input="README.md",      # путь, строка Markdown или base64(Markdown)
            processor=processor,
            processor_kwargs={"add_passage_prefix": False},
        )

        print(result)

asyncio.run(main())
```

### 4) Синхронная загрузка Markdown

```python
from qdrant_vector_storage import QdrantSyncClient, Distance

with QdrantSyncClient(url="http://localhost:6333") as client:
    client.create_collection(
        collection_name="docs",
        vector_size=1024,
        distance=Distance.COSINE,
    )

    result = client.upload_markdown(
        collection_name="docs",
        md_input="README.md",
        processor=processor,
        processor_kwargs={"add_passage_prefix": False},
    )

    print(result)
```

---

## Поддерживаемые форматы входа для MarkdownProcessor

`MarkdownProcessor.build_chunks(...)` принимает:

- строку Markdown
- строку `base64(Markdown)` (авто-распознавание)
- путь к файлу `.md`

---

## Требования к embedder

`MarkdownProcessor` ожидает объект, у которого есть метод:

- `embed(texts: List[str]) -> Iterable[np.ndarray]`

Именно так работает `fastembed.TextEmbedding`.

---

# Документация API

Ниже перечислены все публичные классы и методы.

## 1) MarkdownProcessor

`MarkdownProcessor` выполняет:
- загрузку Markdown (строка / base64 / путь)
- нормализацию
- разбиение на чанки
- расчёт эмбеддингов через переданный `embedder`

### Методы MarkdownProcessor

| Метод | Входные параметры | Выход | Назначение |
|---|---|---|---|
| `MarkdownProcessor(embedder, chunk_size=900, chunk_overlap=120, keep_headings=True, keep_code_blocks=True, passage_prefix="passage: ", batch_size=64, expected_dim=None)` | `embedder`: объект с `.embed(List[str])`; параметры чанкинга/батчинга | `MarkdownProcessor` | Создание процессора Markdown |
| `build_chunks(source, source_name=None, assume_base64_if_looks_like=True, add_passage_prefix=True)` | `source`: `str` или `PathLike` | `List[TextChunk]` (вектор заполнен) | Полный пайплайн: загрузка → чанки → эмбеддинги |
| `embed_query(query_text, add_query_prefix=True)` | `query_text: str` | `List[float]` | Эмбеддинг запроса (для E5-подобных моделей можно добавлять префикс `query:`) |

---

## 2) QdrantSyncClient

Синхронный клиент для Qdrant.

### Методы QdrantSyncClient

| Метод | Входные параметры | Выход | Возможные исключения |
|---|---|---|---|
| `QdrantSyncClient(url, api_key=None, timeout=60, **kwargs)` | `url: str`, `api_key: Optional[str]` | объект клиента | `ConnectionError` |
| `create_collection(collection_name, vector_size, distance=Distance.COSINE, on_disk_payload=True, **kwargs)` | `collection_name: str`, `vector_size: int` | `Dict[str, Any]` | `CollectionExistsError`, `QdrantError` |
| `get_collection_info(collection_name)` | `collection_name: str` | `Dict[str, Any]` | `CollectionNotFoundError`, `QdrantError` |
| `list_collections()` | — | `List[str]` | `QdrantError` |
| `delete_collection(collection_name)` | `collection_name: str` | `bool` | — |
| `upload_points(collection_name, points, batch_size=100, wait=True)` | `points: List[Point]` | `List[str]` (IDs) | `CollectionNotFoundError`, `QdrantError` |
| `upload_markdown(collection_name, md_input, processor, source_name=None, metadata=None, batch_size=100, wait=True, processor_kwargs=None)` | `md_input: str | PathLike`, `processor: MarkdownProcessor` | `FileUploadResult` | `CollectionNotFoundError`, `FileProcessingError`, `EmbeddingError`, `QdrantError` |
| `delete_points(collection_name, point_ids=None, filter_condition=None, wait=True)` | ids или filter | `int` (удалено, если доступно) | `CollectionNotFoundError`, `QdrantError` |
| `delete_by_metadata(collection_name, metadata_key, metadata_value, wait=True)` | ключ/значение | `int` | `CollectionNotFoundError`, `QdrantError` |
| `search(collection_name, query_vector, limit=10, score_threshold=None, filter_condition=None, with_payload=True, with_vectors=False)` | `query_vector: List[float]` | `List[SearchResult]` | `CollectionNotFoundError`, `QdrantError` |
| `count_points(collection_name, filter_condition=None, exact=False)` | фильтр | `int` | `CollectionNotFoundError`, `QdrantError` |
| `healthcheck()` | — | `bool` | — |
| `close()` | — | `None` | — |

---

## 3) QdrantAsyncClient

Асинхронный клиент для Qdrant.

### Методы QdrantAsyncClient

| Метод | Входные параметры | Выход | Возможные исключения |
|---|---|---|---|
| `QdrantAsyncClient(url, api_key=None, timeout=60, **kwargs)` | `url: str`, `api_key: Optional[str]` | объект клиента | `ConnectionError` |
| `create_collection(collection_name, vector_size, distance=Distance.COSINE, on_disk_payload=True, **kwargs)` | `collection_name: str`, `vector_size: int` | `Dict[str, Any]` | `CollectionExistsError`, `QdrantError` |
| `get_collection_info(collection_name)` | `collection_name: str` | `Dict[str, Any]` | `CollectionNotFoundError`, `QdrantError` |
| `list_collections()` | — | `List[str]` | `QdrantError` |
| `delete_collection(collection_name)` | `collection_name: str` | `bool` | — |
| `upload_points(collection_name, points, batch_size=100, wait=True)` | `points: List[Point]` | `List[str]` (IDs) | `CollectionNotFoundError`, `QdrantError` |
| `upload_markdown(collection_name, md_input, processor, source_name=None, metadata=None, batch_size=100, wait=True, processor_kwargs=None)` | `md_input: str | PathLike`, `processor: MarkdownProcessor` | `FileUploadResult` | `CollectionNotFoundError`, `FileProcessingError`, `EmbeddingError`, `QdrantError` |
| `delete_points(collection_name, point_ids=None, filter_condition=None, wait=True)` | ids или filter | `int` (удалено, если доступно) | `CollectionNotFoundError`, `QdrantError` |
| `delete_by_metadata(collection_name, metadata_key, metadata_value, wait=True)` | ключ/значение | `int` | `CollectionNotFoundError`, `QdrantError` |
| `search(collection_name, query_vector, limit=10, score_threshold=None, filter_condition=None, with_payload=True, with_vectors=False)` | `query_vector: List[float]` | `List[SearchResult]` | `CollectionNotFoundError`, `QdrantError` |
| `count_points(collection_name, filter_condition=None, exact=False)` | фильтр | `int` | `CollectionNotFoundError`, `QdrantError` |
| `healthcheck()` | — | `bool` | — |
| `close()` | — | `None` | — |

---

## 4) Утилиты

### FilterBuilder

| Метод | Входные параметры | Выход | Назначение |
|---|---|---|---|
| `FilterBuilder.build_filter(condition)` | `condition: Dict[str, Any]` | `models.Filter | None` | Сборка Qdrant-фильтра из словаря |

### Конвертеры

| Функция | Входные параметры | Выход | Назначение |
|---|---|---|---|
| `chunks_to_points(chunks, base_metadata=None, id_factory=None)` | `List[TextChunk]` | `List[Point]` | Преобразование чанков в точки для upsert |

---

## Лицензия

MIT (см. файл `LICENSE`).
