import re
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from fastembed import TextEmbedding
from typing import List, Optional, Dict, Any, Union, Literal

from .models import EmbeddingModel, TextChunk


class MarkdownProcessor:
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        embedding_model: Union[EmbeddingModel, str] = EmbeddingModel.INTFLOAT_MULTILINGUAL_E5_SMALL,
    ):
        """
        Initialize Markdown processor with embedding support.
        
        Args:
            config: Configuration dictionary for chunking
            embedding_model: Model to use for embeddings (intfloat/multilingual-e5-*)
        """
        self.config = config or {
            "chunk_size": 500,      # максимальный размер чанка в символах
            "chunk_overlap": 50,     # перекрытие между чанками
            "min_chunk_size": 100,   # минимальный размер чанка
            "split_by_headers": True, # разбивать по заголовкам
            "batch_size": 32,        # размер батча для эмбеддингов
        }
        
        # Настройка модели эмбеддингов
        if isinstance(embedding_model, str):
            self.embedding_model = EmbeddingModel(embedding_model)
        else:
            self.embedding_model = embedding_model
        
        self.embedding_service = None
        self._init_embedding_service()
        
        self._init_processing()
    
    def _init_embedding_service(self):
        """Инициализация сервиса эмбеддингов через FastEmbed"""
        try:
            self.embedding_service = TextEmbedding(model_name=self.embedding_model.value)
        except ImportError:
            raise ImportError("Please install fastembed: pip install fastembed")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def set_embedding_model(self, model: Union[EmbeddingModel, str]):
        """Change embedding model"""
        if isinstance(model, str):
            self.embedding_model = EmbeddingModel(model)
        else:
            self.embedding_model = model
        
        self._init_embedding_service()
        return self
    
    def _init_processing(self):
        """Инициализация/сброс состояния"""
        self.raw_text = None
        self.parsed_structure = []  # список блоков
        self.chunks = []             # промежуточные чанки
        self.final_chunks = []       # готовые TextChunk
        self.source_info = {
            "source": "unknown",
            "load_time": datetime.now().isoformat(),
            "embedding_model": self.embedding_model.value,
            "embedding_dimension": self.embedding_model.dimension
        }
    
    # Методы загрузки
    def load_from_file(self, file_path: str) -> 'MarkdownProcessor':
        """Загрузка из файла"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()
        
        self.source_info.update({
            "source": str(path),
            "filename": path.name,
            "file_size": path.stat().st_size,
            "source_type": "file"
        })
        
        return self
    
    def load_from_base64(self, file_base64: Union[str, bytes]) -> 'MarkdownProcessor':
        """Загрузка из base64 строки"""
        try:
            if isinstance(file_base64, str):
                decoded_bytes = base64.b64decode(file_base64)
            else:
                decoded_bytes = file_base64
            
            self.raw_text = decoded_bytes.decode('utf-8')
            
            self.source_info.update({
                "source": "base64_input",
                "source_type": "base64",
                "original_size": len(file_base64) if isinstance(file_base64, str) else len(file_base64)
            })
        except Exception as e:
            raise ValueError(f"Ошибка декодирования base64: {e}")
        
        return self
    
    def load_from_text(self, text: str, source_name: str = "text_input") -> 'MarkdownProcessor':
        """Загрузка из текстовой строки"""
        self.raw_text = text
        self.source_info.update({
            "source": source_name,
            "source_type": "direct_text",
            "text_length": len(text)
        })
        
        return self
    
    # Методы обработки
    def parse(self) -> 'MarkdownProcessor':
        """Парсинг Markdown и создание структурированных блоков"""
        if not self.raw_text:
            raise ValueError("Нет текста для парсинга. Сначала загрузите данные.")
        
        lines = self.raw_text.split('\n')
        current_block = {
            "type": "text",
            "content": "",
            "headers": [],
            "start_line": 0
        }
        
        in_code_block = False
        code_block_lang = None
        
        for i, line in enumerate(lines):
            # Проверка на начало/конец блока кода
            if line.startswith('```'):
                if not in_code_block:
                    # Начало блока кода
                    if current_block["content"].strip():
                        self.parsed_structure.append(current_block)
                    
                    in_code_block = True
                    code_block_lang = line[3:].strip() or "text"
                    current_block = {
                        "type": "code",
                        "language": code_block_lang,
                        "content": "",
                        "headers": [],
                        "start_line": i
                    }
                else:
                    # Конец блока кода
                    current_block["content"] = current_block["content"].rstrip()
                    self.parsed_structure.append(current_block)
                    in_code_block = False
                    current_block = {
                        "type": "text",
                        "content": "",
                        "headers": [],
                        "start_line": i + 1
                    }
                continue
            
            if in_code_block:
                current_block["content"] += line + "\n"
                continue
            
            # Проверка на заголовок
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                # Сохраняем предыдущий блок
                if current_block["content"].strip():
                    self.parsed_structure.append(current_block)
                
                # Создаем новый блок-заголовок
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_block = {
                    "type": "header",
                    "level": level,
                    "content": title,
                    "raw": line,
                    "headers": [{"level": level, "title": title}],
                    "start_line": i
                }
                self.parsed_structure.append(current_block)
                
                # Новый блок для текста после заголовка
                current_block = {
                    "type": "text",
                    "content": "",
                    "headers": [{"level": level, "title": title}],
                    "start_line": i + 1
                }
            
            # Проверка на список
            elif re.match(r'^[\*\-\+]\s+', line.strip()) or re.match(r'^\d+\.\s+', line.strip()):
                if current_block["type"] != "list":
                    if current_block["content"].strip():
                        self.parsed_structure.append(current_block)
                    current_block = {
                        "type": "list",
                        "content": line + "\n",
                        "headers": current_block.get("headers", []),
                        "start_line": i
                    }
                else:
                    current_block["content"] += line + "\n"
            
            # Обычный текст
            else:
                if current_block["type"] not in ["text", "list"]:
                    if current_block["content"].strip():
                        self.parsed_structure.append(current_block)
                    current_block = {
                        "type": "text",
                        "content": line + "\n",
                        "headers": [],
                        "start_line": i
                    }
                else:
                    current_block["content"] += line + "\n"
        
        # Добавляем последний блок
        if current_block["content"].strip():
            self.parsed_structure.append(current_block)
        
        return self
    
    def split(self) -> 'MarkdownProcessor':
        """Разбиение на чанки с учетом структуры и размера"""
        if not self.parsed_structure:
            raise ValueError("Нет структуры для разбиения. Сначала вызовите parse().")
        
        chunk_size = self.config["chunk_size"]
        chunk_overlap = self.config["chunk_overlap"]
        min_chunk_size = self.config["min_chunk_size"]
        
        current_chunk = {
            "text": "",
            "headers": [],
            "blocks": [],
            "start_block": 0
        }
        current_size = 0
        
        for i, block in enumerate(self.parsed_structure):
            block_text = block["content"]
            block_size = len(block_text)
            
            # Блоки кода всегда отдельно
            if block["type"] == "code" and block_size > 0:
                if current_chunk["text"]:
                    self.chunks.append(current_chunk)
                
                self.chunks.append({
                    "text": f"```{block.get('language', '')}\n{block_text}```",
                    "headers": block.get("headers", []),
                    "blocks": [block],
                    "type": "code"
                })
                current_chunk = {
                    "text": "",
                    "headers": [],
                    "blocks": [],
                    "start_block": i + 1
                }
                current_size = 0
                continue
            
            # Если блок слишком большой, разбиваем его
            if block_size > chunk_size:
                if current_chunk["text"]:
                    self.chunks.append(current_chunk)
                
                # Разбиваем большой блок
                parts = self._split_text(block_text, chunk_size, chunk_overlap)
                for j, part in enumerate(parts):
                    self.chunks.append({
                        "text": part,
                        "headers": block.get("headers", []),
                        "blocks": [block],
                        "part": j,
                        "type": "text"
                    })
                
                current_chunk = {
                    "text": "",
                    "headers": [],
                    "blocks": [],
                    "start_block": i + 1
                }
                current_size = 0
                continue
            
            # Если текущий чанк + новый блок превышают размер
            if current_size + block_size > chunk_size and current_chunk["text"]:
                # Проверяем, что чанк не слишком маленький
                if current_size >= min_chunk_size:
                    self.chunks.append(current_chunk)
                
                # Начинаем новый чанк с перекрытием
                overlap_text = ""
                if chunk_overlap > 0 and current_chunk["text"]:
                    # Берем последние chunk_overlap символов из предыдущего чанка
                    overlap_text = current_chunk["text"][-chunk_overlap:]
                
                current_chunk = {
                    "text": overlap_text + block_text,
                    "headers": block.get("headers", []),
                    "blocks": [block],
                    "start_block": i
                }
                current_size = len(current_chunk["text"])
            else:
                # Добавляем блок к текущему чанку
                if current_chunk["text"]:
                    current_chunk["text"] += "\n\n" + block_text
                else:
                    current_chunk["text"] = block_text
                
                # Обновляем заголовки
                if block.get("headers"):
                    current_chunk["headers"].extend(block["headers"])
                
                current_chunk["blocks"].append(block)
                current_size += block_size + 2  # +2 за разделитель
        
        # Добавляем последний чанк
        if current_chunk["text"] and current_size >= min_chunk_size:
            self.chunks.append(current_chunk)
        
        return self
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Вспомогательный метод для разбиения длинного текста"""
        parts = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Ищем границу предложения или абзаца
            if end < text_len:
                # Пробуем найти конец предложения
                last_period = text.rfind('. ', start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
                else:
                    # Ищем границу слова
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        end = last_space
            
            parts.append(text[start:end].strip())
            start = end - overlap if overlap > 0 else end
        
        return parts
    
    def clean(self) -> 'MarkdownProcessor':
        """Очистка текста от Markdown разметки"""
        if not self.chunks:
            # Если split не вызывали, но parse был
            if self.parsed_structure:
                self.split()
            else:
                raise ValueError("Нет данных для очистки")
        
        for chunk in self.chunks:
            text = chunk["text"]
            
            # Сохраняем код-блоки без изменений
            if chunk.get("type") == "code":
                chunk["cleaned_text"] = text
                continue
            
            # Удаляем markdown синтаксис
            # Изображения
            text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
            
            # Ссылки (оставляем только текст)
            text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
            
            # Жирный текст **text** или __text__
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            text = re.sub(r'__(.*?)__', r'\1', text)
            
            # Курсив *text* или _text_
            text = re.sub(r'\*(.*?)\*', r'\1', text)
            text = re.sub(r'_(.*?)_', r'\1', text)
            
            # Код `code`
            text = re.sub(r'`(.*?)`', r'\1', text)
            
            # Блоки кода ```code```
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            
            # HTML теги
            text = re.sub(r'<[^>]+>', '', text)
            
            # Убираем множественные пустые строки
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            # Убираем пробелы в начале и конце строк
            text = '\n'.join(line.strip() for line in text.split('\n'))
            
            # Финальная очистка
            text = text.strip()
            
            chunk["cleaned_text"] = text
        
        return self
    
    def enrich(self) -> 'MarkdownProcessor':
        """Добавление метаданных"""
        if not self.chunks:
            raise ValueError("Нет чанков для обогащения метаданными")
        
        for idx, chunk in enumerate(self.chunks):
            # Берем очищенный текст или оригинал
            text = chunk.get("cleaned_text", chunk["text"])
            
            if not text.strip():
                continue  # Пропускаем пустые чанки
            
            # Собираем все заголовки из блока
            all_headers = []
            header_path = []
            for h in chunk.get("headers", []):
                if isinstance(h, dict):
                    all_headers.append(h.get("title", str(h)))
                    header_path.append(h.get("title", str(h)))
                else:
                    all_headers.append(str(h))
                    header_path.append(str(h))
            
            # Создаем метаданные
            metadata = {
                "source": self.source_info.get("filename", self.source_info.get("source", "unknown")),
                "source_type": self.source_info.get("source_type", "unknown"),
                "chunk_id": idx,
                "chunk_size": len(text),
                "word_count": len(text.split()),
                "char_count": len(text),
                "headers": all_headers,
                "header_path": " > ".join(header_path) if header_path else None,
                "block_types": list(set(b.get("type") for b in chunk.get("blocks", []))),
                "has_code": any(b.get("type") == "code" for b in chunk.get("blocks", [])),
                "hash": hashlib.md5(text.encode()).hexdigest()[:8],
                "created_at": datetime.now().isoformat(),
                "embedding_model": self.embedding_model.value,
                "embedding_dimension": self.embedding_model.dimension,
                "config": self.config.copy()
            }
            
            # Добавляем информацию о части, если есть
            if "part" in chunk:
                metadata["part"] = chunk["part"]
                metadata["total_parts"] = len([c for c in self.chunks if c.get("blocks") == chunk.get("blocks")])
            
            # Сохраняем промежуточные данные
            chunk["metadata"] = metadata
            chunk["final_text"] = text
        
        return self
    
    def embed(self) -> 'MarkdownProcessor':
        """Создание эмбеддингов для всех чанков с правильными префиксами"""
        if not self.chunks:
            raise ValueError("Нет чанков для создания эмбеддингов")
        
        # Собираем все тексты для эмбеддингов (с префиксом passage: для документов)
        texts = []
        chunk_indices = []
        
        for idx, chunk in enumerate(self.chunks):
            text = chunk.get("final_text", chunk.get("cleaned_text", chunk["text"])).strip()
            if text:
                # E5 модели требуют префикс "passage: " для документов
                texts.append(f"passage: {text}")
                chunk_indices.append(idx)
        
        if not texts:
            raise ValueError("Нет текста для создания эмбеддингов")
        
        # Создаем эмбеддинги батчами через FastEmbed
        batch_size = self.config.get("batch_size", 32)
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # FastEmbed возвращает генератор
            batch_embeddings = list(self.embedding_service.embed(batch_texts))
            all_embeddings.extend(batch_embeddings)
        
        # Добавляем эмбеддинги к чанкам
        for i, chunk_idx in enumerate(chunk_indices):
            self.chunks[chunk_idx]["vector"] = all_embeddings[i].tolist() if hasattr(all_embeddings[i], 'tolist') else all_embeddings[i]
        
        return self
    
    def build(self) -> List[TextChunk]:
        """Сборка финальных TextChunk объектов"""
        self.final_chunks = []
        
        for idx, chunk in enumerate(self.chunks):
            text = chunk.get("final_text", chunk.get("cleaned_text", chunk["text"]))
            
            if not text.strip():
                continue
            
            # Создаем TextChunk
            self.final_chunks.append(TextChunk(
                text=text,
                index=idx,
                metadata=chunk.get("metadata", {}),
                vector=chunk.get("vector")
            ))
        
        return self.final_chunks
    
    # Основные методы для разных типов ввода
    def process_file(
        self, 
        file_path: str,
        with_embedding: bool = True
    ) -> List[TextChunk]:
        """Обработка файла"""
        return (self
            .load_from_file(file_path)
            .parse()
            .split()
            .clean()
            .enrich()
            .embed() if with_embedding else self
            .build())
    
    def process_base64(
        self, 
        file_base64: Union[str, bytes],
        with_embedding: bool = True
    ) -> List[TextChunk]:
        """Обработка base64 строки"""
        return (self
            .load_from_base64(file_base64)
            .parse()
            .split()
            .clean()
            .enrich()
            .embed() if with_embedding else self
            .build())
    
    def process_text(
        self, 
        text: str,
        source_name: str = "text_input",
        with_embedding: bool = True
    ) -> List[TextChunk]:
        """Обработка текстовой строки"""
        return (self
            .load_from_text(text, source_name)
            .parse()
            .split()
            .clean()
            .enrich()
            .embed() if with_embedding else self
            .build())
    
    def process_any(
        self,
        input_data: Union[str, bytes],
        input_type: Literal["file", "base64", "text"] = "file",
        with_embedding: bool = True
    ) -> List[TextChunk]:
        """
        Универсальный метод обработки
        
        Args:
            input_data: данные для обработки
            input_type: тип данных ("file", "base64", "text")
            with_embedding: создавать ли эмбеддинги
        """
        if input_type == "file":
            return self.process_file(str(input_data), with_embedding)
        elif input_type == "base64":
            return self.process_base64(input_data, with_embedding)
        elif input_type == "text":
            return self.process_text(str(input_data), with_embedding=with_embedding)
        else:
            raise ValueError(f"Unknown input type: {input_type}")