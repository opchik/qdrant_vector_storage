import os
import re
import uuid
import base64
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from qdrant_client.http import models
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import TextChunk, ChunkType, ChunkMetadata, ParentBlock


class MarkdownProcessor:
    _BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
    
    def __init__(
        self,
        embedder,
        *,
        child_size: int = 450,
        child_overlap: int = 0,
        parent_target_size: int = 1500,
        keep_headings: bool = True,
        keep_code_blocks: bool = True,
        batch_size: int = 64,
        expected_dim: int = 1024,
        parent_max_size: int = 2000
    ):
        """
        Args:
            embedder: Объект для эмбеддинга (с методом encode/embed)
            child_size: Размер child-чанка для поиска
            child_overlap: Перекрытие между child-чанками
            parent_target_size: Целевой размер parent-блока
            keep_headings: Сохранять заголовки
            keep_code_blocks: Сохранять блоки кода
            batch_size: Размер батча для эмбеддинга
            expected_dim: Ожидаемая размерность эмбеддинга
            parent_max_size: Максимальный размер parent-блока
        """
        self._embedder = embedder
        self._expected_dim = expected_dim
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.parent_target_size = parent_target_size
        self.keep_headings = keep_headings
        self.keep_code_blocks = keep_code_blocks
        self.batch_size = batch_size
        self.parent_max_size = parent_max_size
        
        self._validate_params()
    
    def _validate_params(self):
        """Проверка параметров"""
        if self.child_size <= 0:
            raise ValueError("child_size должен быть > 0")
        if self.child_overlap < 0 or self.child_overlap >= self.child_size:
            raise ValueError("child_overlap должен быть >= 0 и < child_size")
        if self.parent_target_size <= self.child_size:
            raise ValueError("parent_target_size должен быть больше child_size")
    
    # ---------------------------- Public API ----------------------------
    
    def build_chunks(
        self,
        source: Union[str, os.PathLike[str]],
        *,
        source_name: Optional[str] = None,
        assume_base64_if_looks_like: bool = True,
        **kwargs
    ) -> TextChunk:
        md_text, resolved_name = self._load_markdown(
            source,
            source_name=source_name,
            assume_base64_if_looks_like=assume_base64_if_looks_like,
        )
        doc_name = Path(resolved_name).name if Path(resolved_name).exists() else resolved_name
        doc_id = f"{doc_name}_{uuid.uuid4()}"
        elements = self._parse_markdown(md_text)
        parents = self._create_parent_blocks(elements, doc_id, doc_name)
        children = self._create_child_chunks(parents, doc_id, doc_name)
        children = self._embed_chunks(children, **kwargs)
        return children
    
    # ---------------------------- Загрузка ----------------------------
    
    def _load_markdown(
        self,
        source: Union[str, os.PathLike[str]],
        *,
        source_name: Optional[str],
        assume_base64_if_looks_like: bool,
    ) -> Tuple[str, str]:
        if isinstance(source, (Path, os.PathLike)):
            p = Path(source)
            return p.read_text(encoding="utf-8"), source_name or str(p)
        if not isinstance(source, str):
            raise TypeError("source должен быть str или path-like")
        s = source.strip()
        if len(s) < 256:
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
        
    # ---------------------------- Парсинг Markdown ----------------------------
    
    class _ElementType(str, Enum):
        HEADING = "heading"
        PARAGRAPH = "paragraph"
        TABLE = "table"
        CODE = "code"
        BLANK = "blank"
    
    class _ParsedElement(BaseModel):
        type: "_ElementType"
        text: str
        heading_level: Optional[int] = None
        heading_text: Optional[str] = None
        table_data: Optional[Dict[str, Any]] = None
        start_line: int = 0
        end_line: int = 0
        model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _parse_markdown(self, md_text: str) -> List[_ParsedElement]:
        md_text = self._normalize_newlines(md_text)
        lines = md_text.split("\n")
        elements = []
        i = 0
        in_code = False
        code_buf = []
        code_start = 0
        while i < len(lines):
            line = lines[i]
            # Блоки кода
            fence_match = re.match(r"^(\s*)(```+|~~~+)\s*(.*)$", line)
            if fence_match:
                if not in_code:
                    in_code = True
                    code_buf = [line]
                    code_start = i
                else:
                    code_buf.append(line)
                    elements.append(self._ParsedElement(
                        type=self._ElementType.CODE,
                        text="\n".join(code_buf),
                        start_line=code_start,
                        end_line=i
                    ))
                    in_code = False
                    code_buf = []
                i += 1
                continue
            if in_code:
                code_buf.append(line)
                i += 1
                continue
            # Заголовки
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                elements.append(self._ParsedElement(
                    type=self._ElementType.HEADING,
                    text=line,
                    heading_level=level,
                    heading_text=title,
                    start_line=i,
                    end_line=i
                ))
                i += 1
                continue
            # Таблицы
            if '|' in line and i + 1 < len(lines) and '|' in lines[i + 1]:
                table_data = self._extract_table(lines, i)
                if table_data:
                    elements.append(self._ParsedElement(
                        type=self._ElementType.TABLE,
                        text=table_data["raw"],
                        table_data=table_data,
                        start_line=i,
                        end_line=i + table_data["line_count"] - 1
                    ))
                    i += table_data["line_count"]
                    continue
            # Пустые строки
            if line.strip() == "":
                elements.append(self._ParsedElement(
                    type=self._ElementType.BLANK,
                    text="",
                    start_line=i,
                    end_line=i
                ))
                i += 1
                continue
            # Параграфы (собираем несколько строк)
            para_lines = [line]
            para_start = i
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if (next_line.startswith('#') or 
                    ('|' in next_line and i + 1 < len(lines) and '|' in lines[i + 1]) or
                    next_line.strip() == "" or
                    re.match(r"^(\s*)(```+|~~~+)", next_line)):
                    break
                para_lines.append(next_line)
                i += 1
            elements.append(self._ParsedElement(
                type=self._ElementType.PARAGRAPH,
                text="\n".join(para_lines),
                start_line=para_start,
                end_line=i-1
            ))
        return elements
    
    def _extract_table(self, lines: List[str], start_idx: int) -> Optional[Dict[str, Any]]:
        """Извлекает Markdown таблицу"""
        table_lines = []
        i = start_idx
        while i < len(lines) and '|' in lines[i]:
            table_lines.append(lines[i].strip())
            i += 1
        if len(table_lines) < 2:
            return None
        try:
            # Простой парсинг для метаданных
            headers = [h.strip() for h in table_lines[0].split('|') if h.strip()]
            data_lines = table_lines[2:] if len(table_lines) > 2 else []
            
            return {
                "raw": '\n'.join(table_lines),
                "headers": headers,
                "rows": len(data_lines),
                "cols": len(headers),
                "line_count": len(table_lines),
                "has_numerical": False  # Можно определить позже
            }
        except Exception:
            return None
    
    # ---------------------------- Создание Parent-блоков ----------------------------
    
    def _create_parent_blocks(
        self,
        elements: List[_ParsedElement],
        doc_id: str,
        doc_name: str
    ) -> List[ParentBlock]:
        parents = []
        current_block = []
        current_size = 0
        block_start_pos = 0
        block_index = 0
        current_chapter = ""
        current_chapter_index = 0
        for i, elem in enumerate(elements):
            elem_type = self._element_type_to_chunk_type(elem.type)
            elem_size = len(elem.text)
            if elem.type in [self._ElementType.TABLE, self._ElementType.CODE]:
                if current_block:
                    parents.extend(self._split_current_block(
                        current_block, current_size, block_start_pos,
                        doc_id, doc_name, current_chapter, current_chapter_index,
                        block_index
                    ))
                    block_index += len(parents) - (block_index if parents else 0)
                    current_block = []
                    current_size = 0
                parent_id = f"{doc_id}_{block_index}"
                parents.append(ParentBlock(
                    id=parent_id,
                    text=elem.text,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    block_type=elem_type,
                    chapter=current_chapter,
                    chapter_index=current_chapter_index,
                    start_position=i,
                    table_markdown=elem.text if elem.type == self._ElementType.TABLE else None,
                    code_language=None
                ))
                block_index += 1
                block_start_pos = i + 1
                continue
            if elem.type == self._ElementType.HEADING:
                if elem.heading_level == 1:
                    current_chapter = elem.heading_text or f"Chapter {current_chapter_index}"
                    current_chapter_index += 1
            if current_size + elem_size > self.parent_max_size and current_block:
                parents.extend(self._split_current_block(
                    current_block, current_size, block_start_pos,
                    doc_id, doc_name, current_chapter, current_chapter_index - 1,
                    block_index
                ))
                block_index += len(parents) - (block_index if parents else 0)
                current_block = []
                current_size = 0
                block_start_pos = i
            current_block.append(elem)
            current_size += elem_size
        if current_block:
            parents.extend(self._split_current_block(
                current_block, current_size, block_start_pos,
                doc_id, doc_name, current_chapter, current_chapter_index,
                block_index
            ))
        return parents
    
    def _element_type_to_chunk_type(self, elem_type: "_ElementType") -> ChunkType:
        mapping = {
            self._ElementType.PARAGRAPH: ChunkType.TEXT,
            self._ElementType.HEADING: ChunkType.TEXT,
            self._ElementType.TABLE: ChunkType.TABLE,
            self._ElementType.CODE: ChunkType.CODE,
        }
        return mapping.get(elem_type, ChunkType.TEXT)
    
    def _split_current_block(
        self,
        elements: List[_ParsedElement],
        total_size: int,
        start_pos: int,
        doc_id: str,
        doc_name: str,
        chapter: str,
        chapter_index: int,
        start_idx: int
    ) -> List[ParentBlock]:
        if total_size <= self.parent_max_size:
            text = "\n\n".join([e.text for e in elements if e.text.strip()])
            parent_id = f"parent_{doc_id}_{start_idx}"
            return [ParentBlock(
                id=parent_id,
                text=text,
                doc_id=doc_id,
                doc_name=doc_name,
                block_type=ChunkType.TEXT,
                chapter=chapter,
                chapter_index=chapter_index,
                start_position=start_pos
            )]
        blocks = []
        current_text = []
        current_size = 0
        block_part = 0
        for elem in elements:
            elem_size = len(elem.text)
            if current_size + elem_size > self.parent_max_size and current_text:
                parent_id = f"parent_{doc_id}_{start_idx}_{block_part}"
                blocks.append(ParentBlock(
                    id=parent_id,
                    text="\n\n".join(current_text),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    block_type=ChunkType.TEXT,
                    chapter=chapter,
                    chapter_index=chapter_index,
                    start_position=start_pos + block_part
                ))
                block_part += 1
                current_text = [elem.text]
                current_size = elem_size
            else:
                current_text.append(elem.text)
                current_size += elem_size
        if current_text:
            parent_id = f"parent_{doc_id}_{start_idx}_{block_part}"
            blocks.append(ParentBlock(
                id=parent_id,
                text="\n\n".join(current_text),
                doc_id=doc_id,
                doc_name=doc_name,
                block_type=ChunkType.TEXT,
                chapter=chapter,
                chapter_index=chapter_index,
                start_position=start_pos + block_part
            ))
        return blocks
    
    # ---------------------------- Создание Child-чанков ----------------------------
    
    def _create_child_chunks(
        self,
        parents: List[ParentBlock],
        doc_id: str,
        doc_name: str
    ) -> List[TextChunk]:
        children = []
        chunk_position = 0
        for parent in parents:
            if parent.block_type in [ChunkType.TABLE, ChunkType.CODE]:
                child = TextChunk(
                    text=parent.text,
                    metadata= ChunkMetadata(
                        parent_id=parent.id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_type=parent.block_type,
                        position=chunk_position
                    )
                )
                children.append(child)
                chunk_position += 1
                continue
            chunks = self._split_text_into_chunks(parent.text)
            for i, chunk_text in enumerate(chunks):
                child = TextChunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        parent_id=parent.id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        chunk_type=ChunkType.TEXT,
                        position=chunk_position + i
                    )
                )
                children.append(child)
            chunk_position += len(chunks)
        return children
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        if len(text) <= self.child_size:
            return [text]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        for sent in sentences:
            sent_size = len(sent)
            if current_size + sent_size <= self.child_size:
                current_chunk.append(sent)
                current_size += sent_size + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                overlap = self._get_overlap_from_previous(chunks[-1]) if chunks else ""
                if overlap:
                    current_chunk = [overlap]
                    current_size = len(overlap)
                else:
                    current_chunk = []
                    current_size = 0
                current_chunk.append(sent)
                current_size += len(sent)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    
    def _get_overlap_from_previous(self, previous_chunk: str) -> str:
        if self.child_overlap <= 0:
            return ""
        words = previous_chunk.split()
        overlap_words = []
        overlap_size = 0
        for word in reversed(words):
            if overlap_size + len(word) + 1 <= self.child_overlap:
                overlap_words.insert(0, word)
                overlap_size += len(word) + 1
            else:
                break
        return " ".join(overlap_words)
    
    # ---------------------------- Эмбеддинг ----------------------------
    
    def _embed_chunks(
        self,
        chunks: List[TextChunk],
        **kwargs
    ) -> List[TextChunk]:
        if not chunks:
            return chunks
        texts = [chunk.text for chunk in chunks]
        vectors = self._embed(texts, **kwargs)
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector
        return chunks
    
    def _embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        if not texts:
            return []
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            if hasattr(self._embedder, "encode"):
                for vec in self._embedder.encode(batch, **kwargs):
                    if hasattr(vec, "tolist"):
                        out.append(vec.tolist())
                    else:
                        out.append(vec)
            elif hasattr(self._embedder, "embed"):
                for vec in self._embedder.embed(batch, **kwargs):
                    if hasattr(vec, "tolist"):
                        out.append(vec.tolist())
                    else:
                        out.append(vec)
            else:
                raise ValueError("Embedder должен иметь метод encode или embed")
        if out and self._expected_dim and len(out[0]) != self._expected_dim:
            raise ValueError(
                f"Неожиданная размерность эмбеддинга: {len(out[0])} вместо {self._expected_dim}"
            )
        return out
    
    # ---------------------------- Вспомогательные методы ----------------------------
    
    def embed_query(self, query: str, **kwargs) -> List[float]:
        if not query.strip():
            raise ValueError("Запрос не может быть пустым")
        vectors = self._embed([query], **kwargs)
        return vectors[0]


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
