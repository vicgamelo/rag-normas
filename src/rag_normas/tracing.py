"""Traces to genai-lens (OpenTelemetry GenAI-shaped spans), off unless GENAI_LENS_ENDPOINT is set.

A W3C ``traceparent`` from the caller (the agent orchestrator, say) makes the RAG spans part of the
caller's trace, so the cascade shows retrieval and the LLM call under the agent's tool call.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from genai_lens import Lens
from genai_lens.sdk import Span, SpanKind

TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")


class _NoopSpan:
    trace_id: str | None = None

    def set_llm(self, **_: Any) -> _NoopSpan:
        return self

    def set_io(self, **_: Any) -> _NoopSpan:
        return self

    def set_attribute(self, *_: Any) -> _NoopSpan:
        return self


class Tracer:
    def __init__(self, lens: Lens | None) -> None:
        self.lens = lens

    @classmethod
    def from_env(cls, project: str = "rag-normas") -> Tracer:
        endpoint = os.getenv("GENAI_LENS_ENDPOINT")
        if not endpoint:
            return cls(None)
        return cls(Lens(endpoint=endpoint, api_key=os.getenv("GENAI_LENS_API_KEY"), project=project))

    @contextmanager
    def span(
        self, name: str, kind: SpanKind = "other", traceparent: str | None = None
    ) -> Iterator[Span | _NoopSpan]:
        if self.lens is None:
            yield _NoopSpan()
            return
        with self.lens.span(name, kind) as span:
            m = TRACEPARENT.match(traceparent or "")
            if m and span.parent_id is None:
                # Adopt the caller's trace: children created below inherit this trace_id.
                span.trace_id, span.parent_id = m.group(1), m.group(2)
            yield span

    def flush(self) -> None:
        if self.lens is not None:
            self.lens.flush()
