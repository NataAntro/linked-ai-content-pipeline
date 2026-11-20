from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from .models import Draft, ScoredMaterial


@dataclass(frozen=True, slots=True)
class EditorialContext:
    material: ScoredMaterial
    accepted_drafts: tuple[Draft, ...]


@dataclass(frozen=True, slots=True)
class StageResult:
    draft: Draft
    detail: str
    changed: bool = False


@dataclass(frozen=True, slots=True)
class EditorialAuditEntry:
    material_id: str
    stage: str
    decision: str
    detail: str


@dataclass(frozen=True, slots=True)
class EditorialRejection:
    material_id: str
    stage: str
    reason: str


@dataclass(frozen=True, slots=True)
class EditorialReview:
    draft: Draft | None
    audit: tuple[EditorialAuditEntry, ...]
    rejection: EditorialRejection | None = None


class EditorialRejectedError(ValueError):
    pass


class EditorialStage(Protocol):
    @property
    def name(self) -> str: ...

    async def apply(self, draft: Draft, context: EditorialContext) -> StageResult: ...


class DraftRevisionProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def revise(
        self,
        draft: Draft,
        context: EditorialContext,
        rejection: EditorialRejection,
    ) -> Draft: ...


class EditorialChain:
    """Executes editorial stages in order and records every decision."""

    def __init__(
        self,
        stages: Sequence[EditorialStage],
        *,
        reviser: DraftRevisionProvider | None = None,
        max_revision_rounds: int = 0,
    ) -> None:
        if not stages:
            raise ValueError("at least one editorial stage is required")
        if max_revision_rounds < 0:
            raise ValueError("max_revision_rounds must not be negative")
        if reviser is None and max_revision_rounds:
            raise ValueError("reviser is required when revision rounds are enabled")
        if reviser is not None and not max_revision_rounds:
            raise ValueError("max_revision_rounds must be positive when reviser is set")
        self._stages = tuple(stages)
        self._reviser = reviser
        self._max_revision_rounds = max_revision_rounds

    async def review(
        self,
        draft: Draft,
        *,
        material: ScoredMaterial,
        accepted_drafts: tuple[Draft, ...],
    ) -> EditorialReview:
        current = draft
        audit: list[EditorialAuditEntry] = []
        context = EditorialContext(material=material, accepted_drafts=accepted_drafts)
        for revision_round in range(self._max_revision_rounds + 1):
            for stage in self._stages:
                try:
                    result = await stage.apply(current, context)
                except EditorialRejectedError as error:
                    reason = str(error)
                    rejection = EditorialRejection(current.material_id, stage.name, reason)
                    audit.append(
                        EditorialAuditEntry(current.material_id, stage.name, "rejected", reason)
                    )
                    if self._reviser is None or revision_round >= self._max_revision_rounds:
                        return EditorialReview(
                            draft=None,
                            audit=tuple(audit),
                            rejection=rejection,
                        )
                    revised = await self._reviser.revise(current, context, rejection)
                    if revised.material_id != current.material_id:
                        raise ValueError("reviser must preserve material_id") from error
                    current = revised
                    audit.append(
                        EditorialAuditEntry(
                            current.material_id,
                            self._reviser.name,
                            "edited",
                            f"повторная редактура после {stage.name}; раунд {revision_round + 1}",
                        )
                    )
                    break
                current = result.draft
                audit.append(
                    EditorialAuditEntry(
                        current.material_id,
                        stage.name,
                        "edited" if result.changed else "accepted",
                        result.detail,
                    )
                )
            else:
                return EditorialReview(draft=current, audit=tuple(audit))
        raise RuntimeError("editorial revision loop exhausted unexpectedly")


class ClarityEditor:
    name = "clarity_editor"
    _extra_blank_lines = re.compile(r"\n{3,}")
    _horizontal_space = re.compile(r"[ \t]+")
    _space_before_punctuation = re.compile(r"\s+([,.;:!?])")

    @classmethod
    def _normalize(cls, value: str) -> str:
        paragraphs = cls._extra_blank_lines.sub("\n\n", value.strip()).split("\n\n")
        return "\n\n".join(
            cls._space_before_punctuation.sub(
                r"\1",
                cls._horizontal_space.sub(" ", paragraph.replace("\n", " ")).strip(),
            )
            for paragraph in paragraphs
            if paragraph.strip()
        )

    async def apply(self, draft: Draft, context: EditorialContext) -> StageResult:
        del context
        edited = Draft(
            material_id=draft.material_id,
            headline=self._normalize(draft.headline),
            body=self._normalize(draft.body),
        )
        changed = edited != draft
        return StageResult(
            edited,
            "нормализованы заголовок и структура абзацев" if changed else "структура ясна",
            changed,
        )


class ForbiddenPhraseGate:
    name = "anti_slop_gate"

    def __init__(self, phrases: Sequence[str] | None = None) -> None:
        configured = phrases or (
            "революционный прорыв",
            "изменит мир",
            "game changer",
            "в современном быстро меняющемся мире",
            "стоит отметить, что",
            "нельзя не отметить",
            "уникальная возможность",
            "новая эра",
        )
        self._phrases = tuple(phrase.casefold() for phrase in configured)

    async def apply(self, draft: Draft, context: EditorialContext) -> StageResult:
        del context
        searchable = f"{draft.headline} {draft.body}".casefold()
        matched = next((phrase for phrase in self._phrases if phrase in searchable), None)
        if matched is not None:
            raise EditorialRejectedError(f"обнаружена шаблонная формулировка: {matched}")
        return StageResult(draft, "шаблонные формулировки не обнаружены")


class SimilarityGate:
    name = "similarity_gate"

    def __init__(self, threshold: float = 0.88) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self._threshold = threshold

    async def apply(self, draft: Draft, context: EditorialContext) -> StageResult:
        candidate = self._comparison_text(draft)
        highest = max(
            (
                SequenceMatcher(None, candidate, self._comparison_text(previous)).ratio()
                for previous in context.accepted_drafts
            ),
            default=0.0,
        )
        if highest >= self._threshold:
            raise EditorialRejectedError(
                f"сходство с принятым черновиком {highest:.2f} превышает {self._threshold:.2f}"
            )
        return StageResult(draft, f"максимальное сходство: {highest:.2f}")

    @staticmethod
    def _comparison_text(draft: Draft) -> str:
        return f"{draft.headline}\n{draft.body}".casefold().strip()


def default_editorial_chain() -> EditorialChain:
    return EditorialChain((ClarityEditor(), ForbiddenPhraseGate(), SimilarityGate()))
