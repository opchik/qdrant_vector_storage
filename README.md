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

## 2) QdrantSyncClient / 3) QdrantAsyncClient

Синхронный и асинхронный клиенты для Qdrant с идентичным набором методов. Различаются только синтаксисом вызова: синхронные методы vs `async/await`.

| Метод | Входные параметры | Выход | Возможные исключения | Описание |
|-------|-------------------|-------|----------------------|----------|
| `QdrantSyncClient(url, api_key=None, timeout=60, **kwargs)`<br>`QdrantAsyncClient(url, api_key=None, timeout=60, **kwargs)` | `url: str` – адрес Qdrant<br>`api_key: Optional[str]` – ключ API (опционально)<br>`timeout: int` – таймаут запросов (по умолч. 60)<br>`**kwargs` – доп. параметры клиента | объект клиента | `ConnectionError` | Инициализация подключения к Qdrant |
| `create_collection(collection_name, vector_size, distance=Distance.COSINE, on_disk_payload=True, **kwargs)` | `collection_name: str` – имя коллекции<br>`vector_size: int` – размерность векторов<br>`distance: Distance` – метрика расстояния (по умолч. COSINE)<br>`on_disk_payload: bool` – хранить payload на диске (по умолч. True)<br>`**kwargs` – доп. параметры создания | `Dict[str, Any]` – информация о созданной коллекции | `CollectionExistsError`<br>`QdrantError` | Создание новой коллекции |
| `get_collection_info(collection_name)` | `collection_name: str` – имя коллекции | `Dict[str, Any]` – информация о коллекции | `CollectionNotFoundError`<br>`QdrantError` | Получение информации о коллекции |
| `list_collections()` | – | `List[str]` – список имен коллекций | `QdrantError` | Получение списка всех коллекций |
| `delete_collection(collection_name)` | `collection_name: str` – имя коллекции | `bool` – успех операции | – | Удаление коллекции |
| `upload_points(collection_name, points, batch_size=100, wait=True)` | `collection_name: str` – имя коллекции<br>`points: List[Point]` – точки для загрузки<br>`batch_size: int` – размер батча (по умолч. 100)<br>`wait: bool` – ждать завершения (по умолч. True) | `List[str]` – ID загруженных точек | `CollectionNotFoundError`<br>`QdrantError` | Загрузка точек в коллекцию |
| `upload_markdown(collection_name, md_input, processor, source_name=None, metadata=None, batch_size=100, wait=True, processor_kwargs=None)` | `collection_name: str` – имя коллекции<br>`md_input: str | PathLike` – Markdown (текст/base64/путь)<br>`processor: MarkdownProcessor` – процессор для чанков<br>`source_name: Optional[str]` – имя источника<br>`metadata: Optional[Dict]` – метаданные<br>`batch_size: int` – размер батча (по умолч. 100)<br>`wait: bool` – ждать завершения (по умолч. True)<br>`processor_kwargs: Optional[Dict]` – параметры процессора | `FileUploadResult` – результат загрузки | `CollectionNotFoundError`<br>`FileProcessingError`<br>`EmbeddingError`<br>`QdrantError` | Загрузка Markdown с автоматической векторизацией |
| `delete_points(collection_name, point_ids=None, filter_condition=None, wait=True)` | `collection_name: str` – имя коллекции<br>`point_ids: Optional[List[str]]` – ID точек<br>`filter_condition: Optional[Dict]` – фильтр для удаления<br>`wait: bool` – ждать завершения (по умолч. True) | `int` – количество удаленных точек | `CollectionNotFoundError`<br>`QdrantError` | Удаление точек по ID или фильтру |
| `delete_by_metadata(collection_name, metadata_key, metadata_value, wait=True)` | `collection_name: str` – имя коллекции<br>`metadata_key: str` – ключ в metadata<br>`metadata_value: Any` – значение<br>`wait: bool` – ждать завершения (по умолч. True) | `int` – количество удаленных точек | `CollectionNotFoundError`<br>`QdrantError` | Удаление точек по значению в metadata (упрощенный вариант `delete_points`) |
| `search(collection_name, query_vector, limit=10, score_threshold=None, filter_condition=None, with_payload=True, with_vectors=False)` | `collection_name: str` – имя коллекции<br>`query_vector: List[float]` – вектор запроса<br>`limit: int` – лимит результатов (по умолч. 10)<br>`score_threshold: Optional[float]` – порог схожести<br>`filter_condition: Optional[Dict]` – фильтр<br>`with_payload: bool` – загружать payload (по умолч. True)<br>`with_vectors: bool` – загружать векторы (по умолч. False) | `List[SearchResult]` – результаты поиска | `CollectionNotFoundError`<br>`QdrantError` | Поиск похожих векторов |
| `count_points(collection_name, filter_condition=None, exact=False)` | `collection_name: str` – имя коллекции<br>`filter_condition: Optional[Dict]` – фильтр<br>`exact: bool` – точный подсчет (по умолч. False) | `int` – количество точек | `CollectionNotFoundError`<br>`QdrantError` | Подсчет точек в коллекции |
| `healthcheck()` | – | `bool` – доступен ли Qdrant | – | Проверка подключения к Qdrant |
| `close()` | – | `None` | – | Закрытие соединения |


## 3) Утилиты

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
