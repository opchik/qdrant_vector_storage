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
    chunk_size=1024,
    chunk_overlap=300,
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

`MarkdownProcessor` ожидает объект, у которого есть один из методов:

- `embed(texts: List[str]) -> Iterable[np.ndarray]`
- `encode(texts: List[str]) -> Iterable[np.ndarray]`

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
| `MarkdownProcessor()` | `embedder`: объект с `.embed(List[str], **kwargs)` или <br> `.encode(List[str], **kwargs)`;<br>`chunk_size: int = 900`;<br>`chunk_overlap: int = 120`;<br>`keep_headings: bool = True`;<br>`keep_code_blocks: bool = True`;<br>`batch_size: int = 64`;<br>`expected_dim: Optional[int] = None`| `MarkdownProcessor` | Создание процессора Markdown |
| `build_chunks()` | `source: Union[str, PathLike]`;<br> `source_name: Optional[str] = None`;<br>`assume_base64_if_looks_like: bool = True`;<br>`**kwargs`| `List[TextChunk]` (вектор заполнен) | Полный пайплайн: загрузка → чанки → эмбеддинги |
| `embed_query()` | `query_text: str`;<br>`**kwargs` | `List[float]` | Эмбеддинг запроса |

---

## 2) QdrantSyncClient / QdrantAsyncClient

Синхронный и асинхронный клиенты для Qdrant с идентичным набором методов. Различаются только синтаксисом вызова: синхронные методы vs `async/await`.

### Конструктор

| Метод | Входные параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `QdrantSyncClient()`<br>`QdrantAsyncClient()` | `url: str`;<br>`api_key: Optional[str]`;<br>`timeout: int = 60`;<br>`**kwargs`| объект клиента | `ConnectionError` | Инициализация подключения к Qdrant |

### Управление коллекциями

| Метод | Входные  параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `create_collection()` | `collection_name: str`;<br>`vector_size: int`;<br>`distance: Distance = Distance.COSINE`;<br>`on_disk_payload: bool = True`;<br>`**kwargs`| `Dict[str, Any]`| `CollectionExistsError`<br>`QdrantError` | Создание новой коллекции |
| `get_collection_info()` | `collection_name: str`| `Dict[str, Any]` – информация о коллекции | `CollectionNotFoundError`<br>`QdrantError` | Получение информации о коллекции |
| `list_collections()` | – | `List[str]` | `QdrantError` | Получение списка всех коллекций |
| `delete_collection()` | `collection_name: str` | `bool` – успех операции | – | Удаление коллекции |

### Загрузка данных

| Метод | Входные  параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `upload_points()` | `collection_name: str`;<br>`points: List[Point]`;<br>`batch_size: int = 100`;<br>`wait: bool = True` | `List[str]` | `CollectionNotFoundError`<br>`QdrantError` | Загрузка точек в коллекцию |
| `upload_markdown()` | `collection_name: str`;<br>`md_input: Union[str,  PathLike]` – Markdown (текст/base64/путь);<br>`processor: MarkdownProcessor`;<br>`source_name: Optional[str]`;<br>`metadata: Optional[Dict]`;<br>`batch_size: int = 100`;<br>`wait: bool = True`;<br>`processor_kwargs: Optional[Dict] = None` | `FileUploadResult` – результат загрузки | `CollectionNotFoundError`<br>`FileProcessingError`<br>`EmbeddingError`<br>`QdrantError` | Загрузка Markdown с автоматической векторизацией |

### Удаление данных

| Метод | Входные параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `delete_points()` | `collection_name: str`;<br>`point_ids: Optional[List[str]]`;<br>`filter_condition: Optional[Dict]`;<br>`wait: bool = True` | `int` – количество удаленных точек | `CollectionNotFoundError`<br>`QdrantError` | Удаление точек по ID или фильтру |

### Поиск (универсальный метод)

| Метод | Входные параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `search()` | `collection_name: str`;<br>`query_vector: Optional[List[float]]`;<br>`query_point_id: Optional[Union[str, int]]`;<br>`filter_condition: Optional[Dict]`;<br>`limit: int = 10`;<br>`score_threshold: Optional[float] = None`;<br>`with_payload: bool = True`;<br>`search_mode: Literal["vector", "id", "filter", "hybrid"]` | `List[SearchResult]` – результаты поиска | `CollectionNotFoundError`<br>`QdrantError`<br>`ValueError` | Универсальный поиск – векторный, по ID, по фильтру или гибридный |

### Вспомогательные методы

| Метод | Входные параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `count_points()` | `collection_name: str`;<br>`filter_condition: Optional[Dict]`;<br>`exact: bool = False` | `int` – количество точек | `CollectionNotFoundError`<br>`QdrantError` | Подсчет точек в коллекции |
| `healthcheck()` | – | `bool` – доступен ли Qdrant | – | Проверка подключения к Qdrant |
| `close()` | – | `None` | – | Закрытие соединения |

## 3) Режимы поиска в методе `search()`

| Режим | Обязательные параметры | Дополнительные параметры | Описание |
|-------|------------------------|--------------------------|----------|
| `'vector'` | `query_vector` | `filter_condition` | Классический поиск по вектору с опциональной фильтрацией |
| `'id'` | `query_point_id` | `filter_condition` | Поиск точек, похожих на точку с указанным ID |
| `'filter'` | `filter_condition` | – | Поиск только по фильтру без вектора |
| `'hybrid'` | `query_vector` | `filter_condition` | Гибридный поиск с RRF (вектор + фильтр) |

## 4) Утилиты

### FilterBuilder

| Метод | Входные параметры | Выход | Назначение |
|---|---|---|---|
| `FilterBuilder.build_filter()` | `condition: Dict[str, Any]` | `models.Filter | None` | Сборка Qdrant-фильтра из словаря |

### Конвертеры

| Функция | Входные параметры | Выход | Назначение |
|---|---|---|---|
| `chunks_to_points()` | `chunks: List[TextChunk]`;<br>`base_metadata: Optional[Dict[str, Any]] = None`;<br>`id_factory: Optional[callable] = None`| `List[Point]` | Преобразование чанков в точки для upsert |
