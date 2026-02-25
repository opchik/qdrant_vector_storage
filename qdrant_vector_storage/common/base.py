import os
import re
import base64
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from qdrant_client.http import models
from typing import Any, Dict, List, Optional, Sequence, Union

from .models import TextChunk


class MarkdownProcessor:
    """
    Чанкер Markdown + расчёт эмбеддингов через fastembed.

    Вход:
      - строка Markdown
      - base64(Markdown)
      - путь к .md

    Выход:
      - List[TextChunk] (vector заполнен)

    Примечание для E5:
      - Для документов рекомендуется префикс "passage: "
      - Для запросов — "query: "
    """

    _BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")

    def __init__(
        self,
        embedder,
        *,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        keep_headings: bool = True,
        keep_code_blocks: bool = True,
        passage_prefix: str = "passage: ",
        batch_size: int = 64,
        expected_dim: Optional[int] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size должен быть > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap должен быть >= 0 и < chunk_size")
        if batch_size <= 0:
            raise ValueError("batch_size должен быть > 0")

        self._embedder = embedder
        self._expected_dim = expected_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_headings = keep_headings
        self.keep_code_blocks = keep_code_blocks
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

    # ---------------------------- Public API ----------------------------

    def build_chunks(
        self,
        source: Union[str, os.PathLike[str]],
        *,
        source_name: Optional[str] = None,
        assume_base64_if_looks_like: bool = True,
        **kwargs
    ) -> List[TextChunk]:
        """
        Полный пайплайн: загрузка -> чанки -> эмбеддинги -> List[TextChunk] с vector.
        """
        md_text, resolved_name = self._load_markdown(
            source,
            source_name=source_name,
            assume_base64_if_looks_like=assume_base64_if_looks_like,
        )
        source_name = Path(resolved_name)
        if source_name.exists():
            resolved_name = source_name.name
        md_text = self._normalize_newlines(md_text)

        units = self._split_to_units(md_text)
        texts = self._units_to_chunks(units)

        # Подготовка строк для эмбеддинга (E5: passage/query префиксы)
        kept_texts: List[str] = []
        for t in texts:
            tt = t.strip()
            if not tt:
                continue
            kept_texts.append(tt)

        vectors = self._embed(kept_texts, **kwargs)

        chunks: List[TextChunk] = []
        for i, (plain, vec) in enumerate(zip(kept_texts, vectors)):
            chunks.append(
                TextChunk(
                    text=plain,
                    index=i,
                    metadata={
                        "source": resolved_name,
                        "embedding_model": getattr(self._embedder, "model_name", type(self._embedder).__name__),
                        "embedding_dimension": len(vec),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "created_at": datetime.now()
                    },
                    vector=vec,
                )
            )
        return chunks

    def embed_query(self, query_text: str, **kwargs) -> List[float]:
        """
        Утилита для эмбеддинга поискового запроса (E5: "query: ").
        """
        text = query_text.strip()
        if not text:
            raise ValueError("query_text пустой")
        vecs = self._embed([text], **kwargs)
        return vecs[0]

    # ---------------------------- Embedding ----------------------------

    def _embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Считает эмбеддинги батчами через fastembed.
        """
        if not texts:
            return []

        out: List[List[float]] = []

        # fastembed возвращает генератор np.ndarray; конвертируем в list[float]
        # Используем батчирование на нашей стороне для предсказуемости по памяти.
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if hasattr(self._embedder, "embed"):
                for vec in self._embedder.embed(batch, **kwargs):
                    v = vec.tolist()  # np.ndarray -> list[float]
                    out.append(v)
            elif hasattr(self._embedder, "encode"):
                for vec in self._embedder.encode(batch, **kwargs):
                    v = vec.tolist()  # np.ndarray -> list[float]
                    out.append(v)
            else: 
                raise ValueError(
                    f"Отсутствуют методы embed или encode для "
                    f"модели {getattr(self._embedder, "model_name", type(self._embedder).__name__)}"
                )
        # (не обязательно) сверка размерности
        if out and self._expected_dim and len(out[0]) != self._expected_dim:
            # Не падаем всегда: в fastembed иногда модель может иметь иную фактическую размерность.
            # Но для Qdrant полезно явно заметить расхождение.
            raise ValueError(
                f"Неожиданная размерность эмбеддинга: {len(out[0])} вместо {self._expected_dim} "
                f"для модели {getattr(self._embedder, "model_name", type(self._embedder).__name__)}"
            )

        return out

    # ---------------------------- Loading ----------------------------

    def _load_markdown(
        self,
        source: Union[str, os.PathLike[str]],
        *,
        source_name: Optional[str],
        assume_base64_if_looks_like: bool,
    ) -> tuple[str, str]:
        if isinstance(source, (Path, os.PathLike)):
            p = Path(source)
            return p.read_text(encoding="utf-8"), source_name or str(p)

        if not isinstance(source, str):
            raise TypeError("source должен быть str или path-like")

        s = source.strip()

        p = Path(s)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8"), source_name or str(p)

        if assume_base64_if_looks_like and self._looks_like_base64(s):
            decoded = self._try_decode_base64(s)
            if decoded is not None:
                return decoded, source_name or "base64"

        return source, source_name or "inline"

    def _looks_like_base64(self, s: str) -> bool:
        if len(s) < 16:
            return False
        if not self._BASE64_RE.match(s):
            return False
        compact = re.sub(r"\s+", "", s)
        return len(compact) % 4 == 0

    def _try_decode_base64(self, s: str) -> Optional[str]:
        try:
            compact = re.sub(r"\s+", "", s)
            data = base64.b64decode(compact, validate=True)
            return data.decode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    # ---------------------------- Chunking ----------------------------

    @dataclass(frozen=True)
    class _Unit:
        kind: str  # "heading" | "para" | "code" | "blank"
        text: str
        heading_level: Optional[int] = None
        heading_text: Optional[str] = None

    def _split_to_units(self, md: str) -> List[_Unit]:
        lines = md.split("\n")
        units: List[MarkdownProcessor._Unit] = []

        i = 0
        in_code = False
        code_buf: List[str] = []
        para_buf: List[str] = []

        def flush_para() -> None:
            nonlocal para_buf
            if not para_buf:
                return
            text = "\n".join(para_buf).strip()
            if text:
                units.append(self._Unit(kind="para", text=text))
            para_buf = []

        while i < len(lines):
            line = lines[i]

            fence_match = re.match(r"^(\s*)(```+|~~~+)\s*(.*)$", line)
            if fence_match:
                if not in_code:
                    flush_para()
                    in_code = True
                    code_buf = [line]
                else:
                    code_buf.append(line)
                    in_code = False
                    if self.keep_code_blocks:
                        units.append(self._Unit(kind="code", text="\n".join(code_buf).strip()))
                    else:
                        units.append(self._Unit(kind="code", text="[code block omitted]"))
                    code_buf = []
                i += 1
                continue

            if in_code:
                code_buf.append(line)
                i += 1
                continue

            h = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if h:
                flush_para()
                level = len(h.group(1))
                title = h.group(2).strip()
                units.append(self._Unit(kind="heading", text=line.strip(), heading_level=level, heading_text=title))
                i += 1
                continue

            if line.strip() == "":
                flush_para()
                units.append(self._Unit(kind="blank", text=""))
                i += 1
                continue

            para_buf.append(line)
            i += 1

        flush_para()

        if in_code and code_buf:
            if self.keep_code_blocks:
                units.append(self._Unit(kind="code", text="\n".join(code_buf).strip()))
            else:
                units.append(self._Unit(kind="code", text="[code block omitted]"))

        return units

    def _units_to_chunks(self, units: Sequence[_Unit]) -> List[str]:
        sections: List[str] = []
        current_lines: List[str] = []
        heading_stack: List[str] = []

        def flush_section() -> None:
            txt = "\n".join(current_lines).strip()
            if txt:
                sections.append(txt)
            current_lines.clear()

        for u in units:
            if u.kind == "heading":
                flush_section()

                if not self.keep_headings:
                    heading_stack = []
                else:
                    level = u.heading_level or 1
                    while len(heading_stack) >= level:
                        heading_stack.pop()
                    heading_stack.append(u.heading_text or "")

                if self.keep_headings and heading_stack:
                    prefix = " > ".join([h for h in heading_stack if h])
                    if prefix:
                        current_lines.append(f"# {prefix}")
                current_lines.append(u.text)

            elif u.kind == "blank":
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
            else:
                current_lines.append(u.text)

        flush_section()

        chunks: List[str] = []
        for sec in sections:
            chunks.extend(self._split_by_size(sec))

        return [c.strip() for c in chunks if c.strip()]

    def _split_by_size(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        blocks = re.split(r"\n{2,}", text)
        blocks = [b.strip() for b in blocks if b.strip()]

        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        def flush_buf() -> None:
            nonlocal buf, buf_len
            if not buf:
                return
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0

        for b in blocks:
            add_len = len(b) + (2 if buf else 0)

            if buf_len + add_len <= self.chunk_size:
                buf.append(b)
                buf_len += add_len
                continue

            # блок больше chunk_size и буфер пуст
            if not buf and len(b) > self.chunk_size:
                chunks.extend(self._hard_split(b))
                continue

            flush_buf()

            # overlap
            if self.chunk_overlap > 0 and chunks:
                ov = self._take_overlap(chunks[-1], self.chunk_overlap)
                if ov:
                    buf = [ov]
                    buf_len = len(ov)

            # добавляем текущий блок
            if len(b) <= self.chunk_size:
                if buf and (buf_len + len(b) + 2 > self.chunk_size):
                    buf = [b]
                    buf_len = len(b)
                else:
                    if buf:
                        buf_len += 2
                    buf.append(b)
                    buf_len += len(b)
            else:
                if buf:
                    flush_buf()
                chunks.extend(self._hard_split(b))

        flush_buf()
        return chunks

    def _hard_split(self, text: str) -> List[str]:
        lines = text.split("\n")
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        def flush() -> None:
            nonlocal cur, cur_len
            if cur:
                chunks.append("\n".join(cur).strip())
            cur = []
            cur_len = 0

        for line in lines:
            add_len = len(line) + (1 if cur else 0)
            if cur_len + add_len <= self.chunk_size:
                if cur:
                    cur_len += 1
                cur.append(line)
                cur_len += len(line)
            else:
                flush()
                if len(line) > self.chunk_size:
                    chunks.extend(self._split_string_fixed(line, self.chunk_size, self.chunk_overlap))
                else:
                    cur = [line]
                    cur_len = len(line)

        flush()
        return chunks

    def _take_overlap(self, text: str, overlap: int) -> str:
        if overlap <= 0:
            return ""
        if len(text) <= overlap:
            return text

        suffix = text[-overlap:]
        idx = suffix.find("\n")
        if idx != -1 and idx + 1 < len(suffix):
            return suffix[idx + 1 :].strip()
        sp = suffix.find(" ")
        if sp != -1 and sp + 1 < len(suffix):
            return suffix[sp + 1 :].strip()
        return suffix.strip()

    def _split_string_fixed(self, s: str, size: int, overlap: int) -> List[str]:
        res: List[str] = []
        step = max(1, size - max(0, overlap))
        for start in range(0, len(s), step):
            part = s[start : start + size].strip()
            if part:
                res.append(part)
            if start + size >= len(s):
                break
        return res


class FilterBuilder:
    """Build Qdrant filters from various input formats."""
    
    # Префикс для полей внутри metadata
    METADATA_PREFIX = "metadata."
    
    @staticmethod
    def build_filter(condition: Union[Dict[str, Any], List, None]) -> Optional[models.Filter]:
        """
        Build Qdrant filter from various input formats.
        
        Поддерживаемые форматы:
        1. Простой словарь: {"chunk_index": 19} -> автоматически добавляет metadata.
        2. Список условий: [{"chunk_index": 19}, {"source": "README.md"}] -> AND между условиями
        3. Полный формат с операторами:
           {
               "must": [
                   {"key": "chunk_index", "match": {"value": 19}},
                   {"key": "score", "range": {"gt": 0.5}}
               ],
               "should": [
                   {"key": "category", "match": {"value": "news"}}
               ],
               "must_not": [
                   {"key": "draft", "match": {"value": True}}
               ]
           }
        4. Прямое field condition: {"key": "chunk_index", "match": {"value": 19}}
        5. Пустой фильтр: None или {}
        
        Args:
            condition: Фильтр в одном из поддерживаемых форматов
            
        Returns:
            Qdrant Filter object или None
        """
        if not condition:
            return None
        
        # Обработка списка условий (AND между элементами)
        if isinstance(condition, list):
            return FilterBuilder._build_from_list(condition)
        
        # Обработка словаря
        if isinstance(condition, dict):
            # Проверяем, это простой key-value словарь?
            if FilterBuilder._is_simple_dict(condition):
                return FilterBuilder._build_from_simple_dict(condition)
            
            # Проверяем, это прямое field condition?
            if "key" in condition:
                return FilterBuilder._build_from_direct_condition(condition)
            
            # Полный формат с операторами
            return FilterBuilder._build_from_operators(condition)
        
        return None
    
    @staticmethod
    def _is_simple_dict(condition: Dict[str, Any]) -> bool:
        """
        Проверяет, является ли словарь простым key-value фильтром.
        Простой словарь не содержит ключей-операторов.
        """
        operator_keys = {'must', 'should', 'must_not', 'key', 'match', 'range', 
                        'geo_radius', 'geo_bounding_box', 'is_null', 'nested'}
        
        # Если есть хотя бы один ключ-оператор - это не простой словарь
        for key in condition.keys():
            if key in operator_keys:
                return False
        
        # Также проверяем, что значения не являются сложными структурами
        # (для простых словарей значения - это скаляры или списки скаляров)
        for value in condition.values():
            if isinstance(value, (dict, list)):
                # Для списков проверяем, что внутри не словари
                if isinstance(value, list) and any(isinstance(item, dict) for item in value):
                    return False
                # Для словарей проверяем, что это не вложенные условия
                if isinstance(value, dict) and any(k in operator_keys for k in value.keys()):
                    return False
        
        return True
    
    @staticmethod
    def _add_metadata_prefix(key: str) -> str:
        """
        Добавляет префикс metadata. к ключу, если его там еще нет.
        """
        if key.startswith("metadata.") or key in ["id", "version"]:
            return key
        return f"{FilterBuilder.METADATA_PREFIX}{key}"
    
    @staticmethod
    def _create_match_condition(key: str, value: Any) -> models.FieldCondition:
        """
        Создает FieldCondition с правильным типом match в зависимости от значения.
        """
        field_path = FilterBuilder._add_metadata_prefix(key)
        
        if isinstance(value, (list, tuple)):
            # Список значений -> MatchAny
            return models.FieldCondition(
                key=field_path,
                match=models.MatchAny(any=list(value))
            )
        elif isinstance(value, str) and len(value.split()) > 1:
            # Длинная строка с пробелами -> MatchText
            return models.FieldCondition(
                key=field_path,
                match=models.MatchText(text=value)
            )
        elif isinstance(value, (int, float, bool, str)):
            # Простые типы -> MatchValue
            return models.FieldCondition(
                key=field_path,
                match=models.MatchValue(value=value)
            )
        elif value is None:
            # None -> IsNull
            return models.FieldCondition(
                key=field_path,
                is_null=models.IsNull(is_null=True)
            )
        else:
            # По умолчанию
            return models.FieldCondition(
                key=field_path,
                match=models.MatchValue(value=value)
            )
    
    @staticmethod
    def _build_from_simple_dict(condition: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Строит фильтр из простого словаря {поле: значение}.
        """
        conditions = []
        for key, value in condition.items():
            field_cond = FilterBuilder._create_match_condition(key, value)
            conditions.append(field_cond)
        
        return models.Filter(must=conditions) if conditions else None
    
    @staticmethod
    def _build_from_list(conditions_list: List) -> Optional[models.Filter]:
        """
        Строит фильтр из списка условий (AND между элементами).
        """
        must_conditions = []
        
        for item in conditions_list:
            if isinstance(item, dict):
                if FilterBuilder._is_simple_dict(item):
                    # Простой словарь в списке
                    for key, value in item.items():
                        field_cond = FilterBuilder._create_match_condition(key, value)
                        must_conditions.append(field_cond)
                elif "key" in item:
                    # Прямое field condition
                    field_cond = FilterBuilder._build_single_field_condition(item)
                    if field_cond:
                        must_conditions.append(field_cond)
        
        return models.Filter(must=must_conditions) if must_conditions else None
    
    @staticmethod
    def _build_from_direct_condition(condition: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Строит фильтр из прямого field condition.
        """
        field_cond = FilterBuilder._build_single_field_condition(condition)
        if field_cond:
            return models.Filter(must=[field_cond])
        return None
    
    @staticmethod
    def _build_single_field_condition(cond: Dict[str, Any]) -> Optional[models.FieldCondition]:
        """
        Строит одно полевое условие из словаря.
        """
        if "key" not in cond:
            return None
        
        # Добавляем префикс metadata к ключу
        key_with_prefix = FilterBuilder._add_metadata_prefix(cond["key"])
        
        # Match conditions
        if "match" in cond:
            match_value = cond["match"]
            if isinstance(match_value, dict):
                if "value" in match_value:
                    return models.FieldCondition(
                        key=key_with_prefix,
                        match=models.MatchValue(value=match_value["value"])
                    )
                elif "text" in match_value:
                    return models.FieldCondition(
                        key=key_with_prefix,
                        match=models.MatchText(text=match_value["text"])
                    )
                elif "any" in match_value:
                    return models.FieldCondition(
                        key=key_with_prefix,
                        match=models.MatchAny(any=match_value["any"])
                    )
                elif "except" in match_value:
                    return models.FieldCondition(
                        key=key_with_prefix,
                        match=models.MatchExcept(except_=match_value["except"])
                    )
        
        # Range conditions
        elif "range" in cond:
            range_params = cond["range"].copy()
            # Конвертируем строки в числа если нужно
            for param in ["gt", "gte", "lt", "lte"]:
                if param in range_params and isinstance(range_params[param], str):
                    try:
                        range_params[param] = float(range_params[param])
                    except ValueError:
                        pass
            return models.FieldCondition(
                key=key_with_prefix,
                range=models.Range(**range_params)
            )
        
        # Geo conditions
        elif "geo_radius" in cond:
            return models.FieldCondition(
                key=key_with_prefix,
                geo_radius=models.GeoRadius(**cond["geo_radius"])
            )
        elif "geo_bounding_box" in cond:
            return models.FieldCondition(
                key=key_with_prefix,
                geo_bounding_box=models.GeoBoundingBox(**cond["geo_bounding_box"])
            )
        
        # Is null
        elif "is_null" in cond:
            return models.FieldCondition(
                key=key_with_prefix,
                is_null=models.IsNull(is_null=cond["is_null"])
            )
        
        return None
    
    @staticmethod
    def _build_from_operators(condition: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Строит фильтр из полного формата с операторами must/should/must_not.
        """
        filter_kwargs = {}
        
        # Must (AND)
        if "must" in condition and isinstance(condition["must"], list):
            must_conditions = []
            for cond in condition["must"]:
                field_cond = FilterBuilder._build_single_field_condition(cond)
                if field_cond:
                    must_conditions.append(field_cond)
            if must_conditions:
                filter_kwargs["must"] = must_conditions
        
        # Should (OR)
        if "should" in condition and isinstance(condition["should"], list):
            should_conditions = []
            for cond in condition["should"]:
                field_cond = FilterBuilder._build_single_field_condition(cond)
                if field_cond:
                    should_conditions.append(field_cond)
            if should_conditions:
                filter_kwargs["should"] = should_conditions
        
        # Must_not (NOT)
        if "must_not" in condition and isinstance(condition["must_not"], list):
            must_not_conditions = []
            for cond in condition["must_not"]:
                field_cond = FilterBuilder._build_single_field_condition(cond)
                if field_cond:
                    must_not_conditions.append(field_cond)
            if must_not_conditions:
                filter_kwargs["must_not"] = must_not_conditions
        
        return models.Filter(**filter_kwargs) if filter_kwargs else None
