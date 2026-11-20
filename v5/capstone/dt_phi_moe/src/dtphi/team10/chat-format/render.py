"""Plain-text stand-in for the Team 10 chat-format `render.py`, for scoring PLAIN replies of another model.

The real `render.py` (teams/10-post-training/chat-format/) needs the Team 3 tokenizer's marker ids, which are not part
of this folder. `scorers.py` only needs it to wrap a generation in a one-turn transcript and read the reply back, so
this stand-in does exactly that for plain text: no tool calls, no reasoning blocks. It is enough for the Indic and
safety row sets (plain replies). It is NOT enough for the tool-call set, whose replies are chat-format tool calls;
that set is not scored here."""
from __future__ import annotations

T = {"endoftext": "<|endoftext|>", "im_start": "<|im_start|>", "im_end": "<|im_end|>"}
EFFORT_PREFIX = "Reasoning effort: "


class RenderError(ValueError):
    """The transcript is not a one-turn plain-text assistant reply."""


def render(messages, effort="1k") -> str:
    raise NotImplementedError("prompt rendering is done with the Phi chat template, not here")


def parse(transcript: str) -> dict:
    marker = f"{T['im_start']}assistant\n"
    if marker not in transcript:
        raise RenderError("no assistant turn")
    body = transcript.split(marker, 1)[1]
    end = f"{T['im_end']}\n{T['endoftext']}"
    if not body.endswith(end):
        raise RenderError("transcript does not end with the closing markers")
    body = body[: -len(end)]
    if T["im_start"] in body or T["im_end"] in body:
        raise RenderError("the reply runs into another turn")
    return {"messages": [{"role": "assistant", "content": body}]}
