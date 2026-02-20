import os
import re
import base64
from pathlib import Path
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
        add_passage_prefix: bool = True,
    ) -> List[TextChunk]:
        """
        Полный пайплайн: загрузка -> чанки -> эмбеддинги -> List[TextChunk] с vector.
        """
        md_text, resolved_name = self._load_markdown(
            source,
            source_name=source_name,
            assume_base64_if_looks_like=assume_base64_if_looks_like,
        )
        md_text = self._normalize_newlines(md_text)

        units = self._split_to_units(md_text)
        texts = self._units_to_chunks(units)

        # Подготовка строк для эмбеддинга (E5: passage/query префиксы)
        embed_texts: List[str] = []
        kept_texts: List[str] = []
        for t in texts:
            tt = t.strip()
            if not tt:
                continue
            kept_texts.append(tt)
            embed_texts.append(f"{self.passage_prefix}{tt}" if add_passage_prefix else tt)

        vectors = self._embed(embed_texts)

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
                        "passage_prefix_added": bool(add_passage_prefix),
                    },
                    vector=vec,
                )
            )
        return chunks

    def embed_query(self, query_text: str, *, add_query_prefix: bool = True) -> List[float]:
        """
        Утилита для эмбеддинга поискового запроса (E5: "query: ").
        """
        q = query_text.strip()
        if not q:
            raise ValueError("query_text пустой")
        text = f"query: {q}" if add_query_prefix else q
        vecs = self._embed([text])
        return vecs[0]

    # ---------------------------- Embedding ----------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
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
            for vec in self._embedder.embed(batch):
                v = vec.tolist()  # np.ndarray -> list[float]
                out.append(v)

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
    """Build Qdrant filters from dict conditions."""
    
    @staticmethod
    def build_filter(condition: Dict[str, Any]):
        """
        Build Qdrant filter from dict.
        
        Args:
            condition: Filter condition dict
            
        Returns:
            Qdrant Filter object
        """
        conditions = []
        if "must" in condition:
            for cond in condition["must"]:
                if "key" in cond and "match" in cond:
                    conditions.append(
                        models.FieldCondition(
                            key=cond["key"],
                            match=models.MatchValue(value=cond["match"]["value"])
                        )
                    )
                elif "key" in cond and "range" in cond:
                    conditions.append(
                        models.FieldCondition(
                            key=cond["key"],
                            range=models.Range(**cond["range"])
                        )
                    )
        
        return models.Filter(must=conditions) if conditions else None