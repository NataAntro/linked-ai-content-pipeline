from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from linked_ai_content_pipeline.editorial import (
    ClarityEditor,
    DraftRevisionProvider,
    EditorialChain,
    EditorialContext,
    EditorialRejection,
    ForbiddenPhraseGate,
    SimilarityGate,
    default_editorial_chain,
)
from linked_ai_content_pipeline.models import (
    Draft,
    Material,
    ScoreBreakdown,
    ScoredMaterial,
)


def _scored_material() -> ScoredMaterial:
    material = Material(
        source="fixture",
        external_id="one",
        title="AI reliability",
        body="Technical source",
        canonical_url="https://example.test/one",
        published_at=datetime(2026, 7, 17, tzinfo=UTC),
        content_fingerprint="one",
    )
    return ScoredMaterial(
        material=material,
        score=0.8,
        reasons=(),
        breakdown=ScoreBreakdown(1.0, 0.8, 0.0, 1.0),
    )


def test_editorial_chain_records_edits_and_rejects_near_duplicates() -> None:
    async def scenario() -> None:
        chain = default_editorial_chain()
        material = _scored_material()
        original = Draft("fixture:one", "  Точный заголовок  ", "Абзац один.\n\n\nАбзац два.")

        accepted = await chain.review(original, material=material, accepted_drafts=())

        assert accepted.draft is not None
        assert accepted.draft.headline == "Точный заголовок"
        assert accepted.draft.body == "Абзац один.\n\nАбзац два."
        assert accepted.audit[0].decision == "edited"

        duplicate = await chain.review(
            accepted.draft,
            material=material,
            accepted_drafts=(accepted.draft,),
        )

        assert duplicate.draft is None
        assert duplicate.rejection is not None
        assert duplicate.rejection.stage == "similarity_gate"
        assert duplicate.audit[-1].decision == "rejected"

    asyncio.run(scenario())


def test_clarity_editor_normalizes_inline_spacing_and_punctuation() -> None:
    async def scenario() -> None:
        review = await default_editorial_chain().review(
            Draft(
                "fixture:one",
                "  Проверяемый   заголовок  ",
                "Первая\tстрока  , продолжение.\nВторая строка !",
            ),
            material=_scored_material(),
            accepted_drafts=(),
        )

        assert review.draft is not None
        assert review.draft.headline == "Проверяемый заголовок"
        assert review.draft.body == "Первая строка, продолжение. Вторая строка!"

    asyncio.run(scenario())


def test_editorial_chain_rejects_template_language_before_similarity_check() -> None:
    async def scenario() -> None:
        review = await default_editorial_chain().review(
            Draft("fixture:one", "Революционный прорыв", "Нейтральный текст"),
            material=_scored_material(),
            accepted_drafts=(),
        )

        assert review.rejection is not None
        assert review.rejection.stage == "anti_slop_gate"
        assert [entry.stage for entry in review.audit] == ["clarity_editor", "anti_slop_gate"]

    asyncio.run(scenario())


def test_editorial_chain_can_revise_and_recheck_a_rejected_draft() -> None:
    class FixtureReviser(DraftRevisionProvider):
        name = "fixture_reviser"

        async def revise(
            self,
            draft: Draft,
            context: EditorialContext,
            rejection: EditorialRejection,
        ) -> Draft:
            del context, rejection
            return Draft(
                draft.material_id,
                draft.headline.replace("Революционный прорыв", "Практическое изменение"),
                draft.body,
            )

    async def scenario() -> None:
        chain = EditorialChain(
            (ClarityEditor(), ForbiddenPhraseGate(), SimilarityGate()),
            reviser=FixtureReviser(),
            max_revision_rounds=1,
        )
        review = await chain.review(
            Draft("fixture:one", "Революционный прорыв", "Нейтральный текст"),
            material=_scored_material(),
            accepted_drafts=(),
        )

        assert review.draft is not None
        assert review.draft.headline == "Практическое изменение"
        assert review.rejection is None
        assert [entry.decision for entry in review.audit].count("rejected") == 1
        assert [entry.stage for entry in review.audit].count("anti_slop_gate") == 2
        assert "fixture_reviser" in [entry.stage for entry in review.audit]

    asyncio.run(scenario())
