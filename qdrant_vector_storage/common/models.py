"""Common models for both sync and async clients."""
import uuid
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Distance(str, Enum):
    """Distance metrics for vector comparison."""
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"


class FileType(str, Enum):
    """Supported file types."""
    MD = "md"


class Point(BaseModel):
    """Point model for upload."""
    id: Optional[str] = Field(default=None, description="Point ID")
    text: Optional[str] = Field(..., description="Point text content", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Point metadata")
    vector: Optional[List[float]] = Field(default=None, description="Point vector")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchResult(BaseModel):
    """Search result model."""
    id: str = Field(..., description="Point ID")
    score: float = Field(..., description="Similarity score", ge=0, le=1)
    text: str = Field(..., description="Point text")
    metadata: Dict[str, Any] = Field(..., description="Point metadata")


class FileUploadResult(BaseModel):
    """File upload result model."""
    file_name: str = Field(..., description="Uploaded file name")
    file_type: FileType = Field(..., description="File type")
    chunks_uploaded: int = Field(..., description="Number of chunks uploaded", ge=0)
    point_ids: List[str] = Field(..., description="Uploaded point IDs")
    collection_name: str = Field(..., description="Collection name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Upload timestamp")


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CODE = "code"


class ChunkMetadata(BaseModel):
    parent_id: str = Field(..., description="ID родительского блока")
    doc_id: str
    doc_name: str
    chunk_type: ChunkType
    position: int = Field(..., description="Порядковый номер в документе")
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="Текст чанка для эмбеддинга (400-500 токенов)")
    vector: Optional[List[float]] = Field(None, description="Вектор эмбеддинга 1024")
    metadata: Optional[ChunkMetadata] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ParentBlock(BaseModel):
    id: str = Field(..., description="Уникальный ID родителя")
    text: str = Field(..., description="Полный текст для LLM (1500-2000 токенов)")
    doc_id: str
    doc_name: str
    block_type: ChunkType
    chapter: str = ""
    chapter_index: int = 0
    start_position: int = Field(..., description="Позиция начала в документе")
    table_markdown: Optional[str] = None 
    code_language: Optional[str] = None 
    headers: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)