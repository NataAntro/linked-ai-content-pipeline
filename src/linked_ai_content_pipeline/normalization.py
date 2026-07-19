from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import Material, RawMaterial

_TRACKING_PARAMETERS = {"fbclid", "gclid", "ref"}
_WHITESPACE = re.compile(r"\s+")


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMETERS
    ]
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            urlencode(sorted(query)),
            "",
        )
    )


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def normalize_material(raw: RawMaterial) -> Material:
    title = normalize_text(raw.title)
    body = normalize_text(raw.body)
    digest_input = f"{title.casefold()}\n{body.casefold()}".encode()
    return Material(
        source=raw.source,
        external_id=raw.external_id,
        title=title,
        body=body,
        canonical_url=canonicalize_url(raw.url),
        published_at=raw.published_at,
        content_fingerprint=hashlib.sha256(digest_input).hexdigest(),
        popularity=raw.popularity,
    )


def normalize_and_deduplicate(materials: Iterable[RawMaterial]) -> tuple[Material, ...]:
    unique: list[Material] = []
    urls: set[str] = set()
    fingerprints: set[str] = set()
    for raw in materials:
        material = normalize_material(raw)
        if material.canonical_url in urls or material.content_fingerprint in fingerprints:
            continue
        urls.add(material.canonical_url)
        fingerprints.add(material.content_fingerprint)
        unique.append(material)
    return tuple(unique)
