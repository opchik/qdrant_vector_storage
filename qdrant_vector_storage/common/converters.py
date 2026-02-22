"""Conversion helpers between intermediate chunks and Qdrant points."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .models import Point, TextChunk


def chunks_to_points(
    chunks: List[TextChunk],
    *,
    base_metadata: Optional[Dict[str, Any]] = None,
    id_factory: Optional[callable] = None,
) -> List[Point]:
    """Convert TextChunk list to Point list for Qdrant upsert.

    Args:
        chunks: List of chunks with computed vectors.
        base_metadata: Metadata merged into each point payload (overridden by chunk.metadata).
        id_factory: Optional callable returning a new id string (default: uuid4).

    Returns:
        List of Points ready for upload.
    """
    base_metadata = dict(base_metadata or {})
    make_id = id_factory or (lambda: str(uuid.uuid4()))

    points: List[Point] = []
    for ch in chunks:
        if ch.vector is None:
            raise ValueError(f"Chunk {ch.index} has no vector")
        if not ch.text:
            continue

        md = dict(base_metadata)
        md.update(ch.metadata or {})

        points.append(
            Point(
                id=make_id(),
                text=ch.text,
                metadata=md,
                vector=ch.vector,
            )
        )
    return points
