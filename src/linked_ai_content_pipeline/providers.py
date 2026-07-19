from __future__ import annotations

from typing import Protocol

from .models import Draft, ScoredMaterial


class DraftProvider(Protocol):
    async def generate(self, material: ScoredMaterial) -> Draft: ...


class DeterministicDraftProvider:
    """Reproducible local provider with no network or credentials."""

    async def generate(self, material: ScoredMaterial) -> Draft:
        source = material.material
        excerpt = source.body[:180].rstrip()
        if len(source.body) > len(excerpt):
            excerpt += "…"
        return Draft(
            material_id=f"{source.source}:{source.external_id}",
            headline=f"Черновик: {source.title}",
            body=f"{excerpt}\n\nИсточник: {source.canonical_url}",
        )
