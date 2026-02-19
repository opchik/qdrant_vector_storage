from ..qdrant_vector_storage import QdrantAsyncClient, TextChunk

import base64
from typing import Optional, List


class MarkdownProcessor:
    def __init__(self):
        self._init_processing()

    def _init_processing(self):
        self.raw_text = None
        self.parsed_structure = None
        self.chunks = None
    
    def _load_from_file(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()
            
    def _load_from_base64(self, file_base64: str):
        decoded_bytes = base64.b64decode(file_base64)
        self.raw_text = decoded_bytes.decode('utf-8')
    
    def _parse(self):
        # Парсинг структуры
        return self
    
    def _split(self):
        # Разбиение на чанки
        return self
    
    def _clean(self):
        # Очистка текста
        return self
    
    def _enrich(self):
        # Добавление метаданных
        return self
    
    def _get_chunks(self):
        # Получение результата
        return self.chunks
    
    def process_file(
        self, 
        file_path: Optional[str] = None, 
        file_base64: Optional[str] = None
    ) -> List[TextChunk]:
        self._init_processing()
        if file_path:
            self._load_from_file(file_path=file_path)
        elif file_base64:
            self._load_from_base64(file_base64=file_base64)
        else:
            raise ValueError("Empty file input. Either file_path or file_base64 must be provided.")
        self._parse()
        self._clean()
        self._enrich()
        self._get_chunks()
        return self.chunks
        
