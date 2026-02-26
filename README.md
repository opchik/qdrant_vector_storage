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
| `MarkdownProcessor(embedder, chunk_size=900, chunk_overlap=120, keep_headings=True, keep_code_blocks=True, batch_size=64, expected_dim=None)` | `embedder`: объект с `.embed(List[str])`; параметры чанкинга/батчинга | `MarkdownProcessor` | Создание процессора Markdown |
| `build_chunks(source, source_name=None, assume_base64_if_looks_like=True, **kwargs)` | `source`: `str` или `PathLike` | `List[TextChunk]` (вектор заполнен) | Полный пайплайн: загрузка → чанки → эмбеддинги |
| `embed_query(query_text, **kwargs)` | `query_text: str` | `List[float]` | Эмбеддинг запроса |

---

## 2) QdrantSyncClient / QdrantAsyncClient

Синхронный и асинхронный клиенты для Qdrant с идентичным набором методов. Различаются только синтаксисом вызова: синхронные методы vs `async/await`.

### Конструктор

| Метод | Параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `QdrantSyncClient()`<br>`QdrantAsyncClient()` | `url: str` – адрес Qdrant<br>`api_key: Optional[str]` – ключ API<br>`timeout: int` – таймаут запросов (60)<br>`**kwargs` – доп. параметры клиента | объект клиента | `ConnectionError` | Инициализация подключения к Qdrant |

### Управление коллекциями

| Метод | Параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `create_collection()` | `collection_name: str` – имя коллекции<br>`vector_size: int` – размерность векторов<br>`distance: Distance` – метрика (COSINE)<br>`on_disk_payload: bool` – payload на диске (True)<br>`**kwargs` – доп. параметры | `Dict[str, Any]` – информация о коллекции | `CollectionExistsError`<br>`QdrantError` | Создание новой коллекции |
| `get_collection_info()` | `collection_name: str` – имя коллекции | `Dict[str, Any]` – информация о коллекции | `CollectionNotFoundError`<br>`QdrantError` | Получение информации о коллекции |
| `list_collections()` | – | `List[str]` – список имен коллекций | `QdrantError` | Получение списка всех коллекций |
| `delete_collection()` | `collection_name: str` – имя коллекции | `bool` – успех операции | – | Удаление коллекции |

### Загрузка данных

| Метод | Параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `upload_points()` | `collection_name: str` – имя коллекции<br>`points: List[Point]` – точки для загрузки<br>`batch_size: int` – размер батча (100)<br>`wait: bool` – ждать завершения (True) | `List[str]` – ID загруженных точек | `CollectionNotFoundError`<br>`QdrantError` | Загрузка точек в коллекцию |
| `upload_markdown()` | `collection_name: str` – имя коллекции<br>`md_input: str | PathLike` – Markdown (текст/base64/путь)<br>`processor: MarkdownProcessor` – процессор для чанков<br>`source_name: Optional[str]` – имя источника<br>`metadata: Optional[Dict]` – метаданные<br>`batch_size: int` – размер батча (100)<br>`wait: bool` – ждать завершения (True)<br>`processor_kwargs: Optional[Dict]` – параметры процессора | `FileUploadResult` – результат загрузки | `CollectionNotFoundError`<br>`FileProcessingError`<br>`EmbeddingError`<br>`QdrantError` | Загрузка Markdown с автоматической векторизацией |

### Удаление данных

| Метод | Параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `delete_points()` | `collection_name: str` – имя коллекции<br>`point_ids: Optional[List[str]]` – ID точек<br>`filter_condition: Optional[Dict]` – фильтр для удаления<br>`wait: bool` – ждать завершения (True) | `int` – количество удаленных точек | `CollectionNotFoundError`<br>`QdrantError` | Удаление точек по ID или фильтру |
| `delete_by_metadata()` | `collection_name: str` – имя коллекции<br>`metadata_key: str` – ключ в metadata<br>`metadata_value: Any` – значение<br>`wait: bool` – ждать завершения (True) | `int` – количество удаленных точек | `CollectionNotFoundError`<br>`QdrantError` | Удаление точек по значению в metadata |

### Поиск (универсальный метод)

| Метод | Параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `search()` | `collection_name: str` – имя коллекции<br>`query_vector: Optional[List[float]]` – вектор запроса<br>`query_point_id: Optional[Union[str, int]]` – ID точки для поиска похожих<br>`filter_condition: Optional[Dict]` – фильтр<br>`limit: int` – лимит результатов (10)<br>`score_threshold: Optional[float]` – порог схожести<br>`with_payload: bool` – загружать payload (True)<br>`search_mode: Literal["vector", "id", "filter", "hybrid"]` – режим поиска ("vector") | `List[SearchResult]` – результаты поиска | `CollectionNotFoundError`<br>`QdrantError`<br>`ValueError` | Универсальный поиск – векторный, по ID, по фильтру или гибридный |

### Вспомогательные методы

| Метод | Параметры | Выход | Исключения | Описание |
|-------|-----------|-------|------------|----------|
| `count_points()` | `collection_name: str` – имя коллекции<br>`filter_condition: Optional[Dict]` – фильтр<br>`exact: bool` – точный подсчет (False) | `int` – количество точек | `CollectionNotFoundError`<br>`QdrantError` | Подсчет точек в коллекции |
| `healthcheck()` | – | `bool` – доступен ли Qdrant | – | Проверка подключения к Qdrant |
| `close()` | – | `None` | – | Закрытие соединения |

## 3) Режимы поиска в методе `search()`

| Режим | Обязательные параметры | Дополнительные параметры | Описание |
|-------|------------------------|--------------------------|----------|
| `"vector"` | `query_vector` | `filter_condition` | Классический поиск по вектору с опциональной фильтрацией |
| `"id"` | `query_point_id` | `filter_condition` | Поиск точек, похожих на точку с указанным ID |
| `"filter"` | `filter_only` | – | Поиск только по фильтру без вектора |
| `"hybrid"` | `query_vector` | `filter_condition` | Гибридный поиск с RRF (вектор + фильтр) |

## 4) Утилиты

### FilterBuilder

| Метод | Входные параметры | Выход | Назначение |
|---|---|---|---|
| `FilterBuilder.build_filter(condition)` | `condition: Dict[str, Any]` | `models.Filter | None` | Сборка Qdrant-фильтра из словаря |

### Конвертеры

| Функция | Входные параметры | Выход | Назначение |
|---|---|---|---|
| `chunks_to_points(chunks, base_metadata=None, id_factory=None)` | `List[TextChunk]` | `List[Point]` | Преобразование чанков в точки для upsert |
