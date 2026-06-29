"""Shared extraction logic for both model tiers.

Both tiers use an **identical prompt** and structured-output schema — only the
model id differs — so that F1 differences are attributable to model capability,
not prompting (a requirement from the experimental design).

The model reads the document's OCR text (reconstructed from word boxes in
reading order) and returns a flat list of {label, value} field extractions.
Responses are cached per (tier, doc_id) to ``data/processed/extraction_cache/``
because API calls are slow and cost money.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.data.base_loader import Document

_CACHE_ROOT = Path("data/processed/extraction_cache")

_SYSTEM_BASE = (
    "You are a document field-extraction engine. You are given the OCR text of a "
    "single document (a receipt or form). "
)

_SYSTEM_SCHEMA = (
    "Extract ONLY the fields listed below, using EXACTLY these label names. "
    "Use values exactly as they appear in the text. "
    "If a field is not present in the document, omit it.\n\nTarget fields:\n{schema}"
)

_SYSTEM_FREE = (
    "Extract every field as a flat list of {{label, value}} pairs. "
    "Use values exactly as they appear in the text. "
    "Do not invent fields that are not present."
)


def _build_system(schema: dict[str, str] | None) -> str:
    if schema:
        lines = "\n".join(f"  - {label}: {desc}" for label, desc in schema.items())
        return _SYSTEM_BASE + _SYSTEM_SCHEMA.format(schema=lines)
    return _SYSTEM_BASE + _SYSTEM_FREE

# Tool-use forces structured JSON output via the Anthropic SDK.
# tool_choice={"type":"tool","name":"extract_fields"} guarantees the model
# always calls the tool (no free-text fallback).
_TOOL = {
    "name": "extract_fields",
    "description": "Return all extracted fields as a flat list of label/value pairs.",
    "input_schema": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["label", "value"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["fields"],
        "additionalProperties": False,
    },
}


@dataclass
class ExtractionResult:
    fields: list[dict]       # [{"label","value"}, ...]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False


def reconstruct_text(doc: Document) -> str:
    """Join OCR tokens in reading order (top-to-bottom, left-to-right) into lines."""
    words = [w for w in doc.words if w.text.strip()]
    if not words:
        return ""
    # cluster into lines by y, using a tolerance of ~half the median height
    heights = sorted(w.height for w in words if w.height > 0)
    tol = (heights[len(heights) // 2] / 2) if heights else 8.0
    words.sort(key=lambda w: (w.bbox[1], w.bbox[0]))
    lines: list[list] = []
    for w in words:
        if lines and abs(w.bbox[1] - lines[-1][0].bbox[1]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    out = []
    for line in lines:
        line.sort(key=lambda w: w.bbox[0])
        out.append(" ".join(w.text for w in line))
    return "\n".join(out)


class BaseExtractor:
    """One model tier. Subclasses set ``tier``, ``model_id`` and pricing."""

    tier: str = "base"
    model_id: str = ""
    price_in: float = 0.0   # USD / 1M input tokens
    price_out: float = 0.0  # USD / 1M output tokens
    max_tokens: int = 4096

    def __init__(self, client=None):
        self._client = client  # injected anthropic.Anthropic(); lazy by default

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy import so the scorer/tests don't need the SDK
            self._client = anthropic.Anthropic()
        return self._client

    def extract(
        self,
        doc: Document,
        *,
        use_cache: bool = True,
        schema: dict[str, str] | None = None,
    ) -> ExtractionResult:
        """Extract fields from doc.

        Args:
            schema: target field definitions, e.g.
                    {"company": "business name", "total": "amount paid"}.
                    When provided, the model is constrained to these labels only,
                    making F1 evaluation meaningful. Without it, the model invents
                    its own labels which won't match any ground-truth schema.
        """
        cache_fp = _CACHE_ROOT / self.tier / doc.dataset / doc.split / f"{doc.doc_id}.json"
        if use_cache and cache_fp.exists():
            data = json.loads(cache_fp.read_text())
            return ExtractionResult(**data, cached=True)

        text = reconstruct_text(doc)
        client = self._get_client()
        resp = client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=_build_system(schema),
            messages=[{"role": "user", "content": f"OCR text:\n\n{text}"}],
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "extract_fields"},
        )
        tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
        fields = tool_block.input.get("fields", []) if tool_block else []
        cost = (
            resp.usage.input_tokens * self.price_in
            + resp.usage.output_tokens * self.price_out
        ) / 1_000_000

        result = ExtractionResult(
            fields=fields,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=cost,
        )
        if use_cache:
            cache_fp.parent.mkdir(parents=True, exist_ok=True)
            cache_fp.write_text(json.dumps({
                "fields": result.fields,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
            }))
        return result
