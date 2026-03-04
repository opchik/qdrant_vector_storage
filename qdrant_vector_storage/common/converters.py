from typing import List

from .models import Point, TextChunk


def chunks_to_points(chunks: List[TextChunk]) -> List[Point]:
    points: List[Point] = []
    for ch in chunks:
        if ch.vector is None:
            raise ValueError(f"Chunk {ch.index} has no vector")
        if not ch.text:
            continue
        md = ch.metadata.model_dump()
        points.append(
            Point(
                id=ch.id,
                text=ch.text,
                metadata=md,
                vector=ch.vector,
            )
        )
    return points
