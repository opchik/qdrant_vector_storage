"""Common models for both sync and async clients."""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


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


class TextChunk(BaseModel):
    """Text chunk model."""
    text: str = Field(..., description="Chunk text")
    index: int = Field(..., description="Chunk index", ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    vector: Optional[List[float]] = Field(default=None, description="Point vector")
