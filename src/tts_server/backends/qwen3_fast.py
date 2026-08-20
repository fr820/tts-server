"""Fast replacement for qwen-tts's nested code-predictor ``generate``.

Why this exists (measured on NVIDIA A10, bf16, qwen-tts 0.1.1):
``Qwen3TTSTalkerForConditionalGeneration.forward`` runs a *full HuggingFace
``generate()``* on the 5-layer code predictor once per audio frame
(~12 Hz), each generating 15 code-group tokens. That is ~1000 HF
``_sample`` iterations per request for a model whose per-step GPU compute
is <1 ms — the request is host-bound at ~5.5 ms per forward call (Python
dispatch of dozens of tiny CUDA ops per layer) and RTF sits ~1.46
(identically for the 0.6B and 1.7B talkers, because the cost is this
fixed machinery, not the model).

Two stages, both preserving output semantics (same DynamicCache usage,
same per-position embedding tables and per-position lm-heads, same
temperature/top-k/top-p sampling or greedy argmax):

1. A hand-rolled KV-cached loop over the predictor's own ``forward`` —
   removes the HF-generate bookkeeping per step.
2. That loop captured into a single CUDA graph — every shape in the
   15-step loop is static (batch 1, prefill len 2, one token per step,
   token selection is a GPU->GPU dependency with no host sync), so the
   whole nested generation replays as one kernel launch sequence
   (~76 ms -> a few ms per frame on the A10). If graph capture fails on
   an unsupported op, we fall back to the eager loop (correct, slower).

The contract with the qwen-tts caller is exactly what it consumes:
``.sequences`` of shape ``(batch, num_code_groups - 1)`` — verified by
tracing the stock path (every nested call emits a fixed-length
``(1, 15)``; there is no early stop).

Enabled by default via ``backend.options.fast_subtalker``; set it to
``false`` to restore the stock qwen-tts path (A-B benchmarking hook).
``subtalker_greedy: true`` switches token selection from sampling to
argmax (deterministic; changes decoded audio vs sampling).
"""

from __future__ import annotations

import logging
import types
from typing import Any

logger = logging.getLogger("tts_server.backends.qwen3_fast")


def _select(logits: Any, *, do_sample: bool, temperature: float, top_k: int, top_p: float) -> Any:
    """Next-token id (shape ``[B, 1]``) from last-position logits, mirroring
    HF's warper order: temperature -> top_k -> top_p (top_p == 1.0 is a
    no-op, exactly as in HF ``get_logits_warper``)."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    import torch

    logits = logits / temperature
    if top_k and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p != 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        remove = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


def make_fast_generate(*, do_sample: bool, temperature: float, top_k: int, top_p: float) -> Any:
    """Build a ``generate(inputs_embeds=..., max_new_tokens=N, ...)`` replacement
    closing over the sampling parameters (defaults mirror the stock nested call:
    temperature 0.9, top_k 50, top_p 1.0)."""

    def _cat(seqs: list) -> Any:
        import torch

        return torch.cat(seqs, dim=1)

    def fast_generate(self, inputs_embeds=None, max_new_tokens=15, **kwargs):  # noqa: ANN001, ANN003
        # Prefill (seq len 2: past_hidden + last id). forward() derives
        # cache_position/position_ids and the causal mask on its own.
        out = self.forward(inputs_embeds=inputs_embeds, use_cache=True)
        token = _select(
            out.logits[:, -1],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        seqs = [token]
        gs = out.generation_steps
        past = out.past_key_values
        for _ in range(max_new_tokens - 1):
            out = self.forward(
                input_ids=token,
                generation_steps=gs,
                past_key_values=past,
                use_cache=True,
            )
            token = _select(
                out.logits[:, -1],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            seqs.append(token)
            gs = out.generation_steps
            past = out.past_key_values
        return types.SimpleNamespace(sequences=_cat(seqs))

    return fast_generate


class _GraphedGenerate:
    """Wraps the eager fast loop in a lazily-captured CUDA graph.

    The graph is built on the first call (server warmup normally pays
    this) and replayed afterwards: copy inputs into a static buffer,
    replay, return a namespace over the static output tensor. Capture
    failure (an un-capturable op) downgrades permanently to the eager
    loop with a single warning.
    """

    def __init__(self, predictor: Any, eager: Any) -> None:
        self._predictor = predictor
        self._eager = eager
        self._graph = None
        self._n_tokens: int | None = None
        self._static_in: Any = None
        self._static_out: Any = None

    def _try_capture(self, inputs_embeds: Any) -> bool:
        import torch

        # Warmup iterations on a side stream, per the CUDA graphs recipe.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._eager(self._predictor, inputs_embeds=inputs_embeds)
        torch.cuda.current_stream().wait_stream(side)

        self._static_in = torch.empty_like(inputs_embeds)
        self._static_in.copy_(inputs_embeds)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._static_out = self._eager(
                self._predictor, inputs_embeds=self._static_in
            ).sequences
        self._graph = graph
        return True

    def __call__(self, inputs_embeds=None, max_new_tokens=15, **kwargs):  # noqa: ANN001, ANN003
        if self._graph is None:
            try:
                self._try_capture(inputs_embeds)
                self._n_tokens = max_new_tokens
                logger.info("qwen3 fast sub-talker: CUDA graph captured")
            except Exception:
                logger.warning(
                    "qwen3 fast sub-talker: CUDA graph capture failed; "
                    "using eager fast loop",
                    exc_info=True,
                )
                self._graph = False  # sentinel: permanently eager
        if self._graph is False or max_new_tokens != self._n_tokens:
            # Unexpected length (config drift): stay correct, skip the graph.
            return self._eager(self._predictor, inputs_embeds=inputs_embeds)
        self._static_in.copy_(inputs_embeds)
        self._graph.replay()
        return types.SimpleNamespace(sequences=self._static_out)


def patch_code_predictor(predictor: Any, *, greedy: bool = False) -> None:
    """Install the fast generate on a loaded code predictor (in place).

    ``predictor`` is ``Qwen3TTSModel.model.talker.code_predictor``. The
    instance attribute shadows the class's ``generate``;
    ``unpatch_code_predictor`` removes it to restore the stock path.
    """
    eager = make_fast_generate(
        do_sample=not greedy,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
    )
    # Plain instance attribute (not MethodType): the graph wrapper is itself
    # the bound callable; predictor stays reachable inside it.
    predictor.generate = _GraphedGenerate(predictor, eager)


def unpatch_code_predictor(predictor: Any) -> None:
    """Remove the fast generate installed by ``patch_code_predictor``."""
    predictor.__dict__.pop("generate", None)
