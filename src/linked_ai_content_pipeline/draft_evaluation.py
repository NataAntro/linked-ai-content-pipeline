from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Draft
from .rag import KnowledgeDocument

_URL = re.compile(r"https?://[^\s)\]]+")
_TOKEN = re.compile(r"[\w-]+", re.UNICODE)
_NUMBER_CLAIM = re.compile(
    r"\b(\d+(?:[.,]\d+)?)(?=\s*(?:%|процент|миллион|млрд)(?:\s|[.,!?]|$))",
    re.IGNORECASE,
)
_STOPWORDS = {
    "этот",
    "эта",
    "это",
    "который",
    "черновик",
    "контекст",
    "источник",
    "меняет",
    "помогает",
    "через",
    "после",
}
_DEFAULT_FORBIDDEN = (
    "революционный прорыв",
    "изменит мир",
    "game changer",
    "как языковая модель",
)


@dataclass(frozen=True, slots=True)
class DraftEvaluationPolicy:
    required_terms: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = _DEFAULT_FORBIDDEN
    minimum_grounded_token_ratio: float = 0.55


@dataclass(frozen=True, slots=True)
class DraftEvaluation:
    passed: bool
    grounded_token_ratio: float
    citation_precision: float
    citation_recall: float
    required_term_coverage: float
    unsupported_numbers: tuple[str, ...]
    forbidden_phrases: tuple[str, ...]
    issues: tuple[str, ...]


def evaluate_draft(
    draft: Draft,
    *,
    source_text: str,
    source_url: str,
    context: tuple[KnowledgeDocument, ...],
    policy: DraftEvaluationPolicy | None = None,
) -> DraftEvaluation:
    policy = policy or DraftEvaluationPolicy()
    if not 0.0 <= policy.minimum_grounded_token_ratio <= 1.0:
        raise ValueError("minimum_grounded_token_ratio must be between 0 and 1")
    candidate_text = f"{draft.headline}\n{draft.body}".casefold()
    text_without_urls = _URL.sub(" ", candidate_text)
    evidence_text = " ".join([source_text, *(item.text for item in context)]).casefold()
    candidate_tokens = _meaningful_tokens(text_without_urls)
    evidence_tokens = _meaningful_tokens(evidence_text)
    grounded_token_ratio = (
        1.0
        if not candidate_tokens
        else len(candidate_tokens.intersection(evidence_tokens)) / len(candidate_tokens)
    )

    candidate_urls = set(_URL.findall(candidate_text))
    expected_urls = {item.source_url for item in context}
    allowed_urls = expected_urls | {source_url}
    citation_precision = (
        1.0
        if not candidate_urls
        else len(candidate_urls.intersection(allowed_urls)) / len(candidate_urls)
    )
    citation_recall = (
        1.0
        if not expected_urls
        else len(candidate_urls.intersection(expected_urls)) / len(expected_urls)
    )

    required = tuple(term.casefold() for term in policy.required_terms)
    required_term_coverage = (
        1.0 if not required else sum(term in candidate_text for term in required) / len(required)
    )
    evidence_numbers = set(_NUMBER_CLAIM.findall(evidence_text))
    unsupported_numbers = tuple(
        sorted(set(_NUMBER_CLAIM.findall(text_without_urls)) - evidence_numbers)
    )
    forbidden = tuple(
        phrase for phrase in policy.forbidden_phrases if phrase.casefold() in candidate_text
    )

    issues: list[str] = []
    if grounded_token_ratio < policy.minimum_grounded_token_ratio:
        issues.append("draft contains too many tokens unsupported by evidence")
    if citation_precision < 1.0:
        issues.append("draft contains an unknown citation URL")
    if citation_recall < 1.0:
        issues.append("draft omits an expected context citation")
    if required_term_coverage < 1.0:
        issues.append("draft omits a required concept")
    if unsupported_numbers:
        issues.append("draft contains numeric claims absent from evidence")
    if forbidden:
        issues.append("draft contains a forbidden phrase")
    return DraftEvaluation(
        passed=not issues,
        grounded_token_ratio=grounded_token_ratio,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        required_term_coverage=required_term_coverage,
        unsupported_numbers=unsupported_numbers,
        forbidden_phrases=forbidden,
        issues=tuple(issues),
    )


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(text)
        if len(token) >= 4 and token not in _STOPWORDS and not token.isdigit()
    }
