"""Unit tests for the fast sub-talker generate replacement (qwen3_fast).

These exercise the eager fast loop with a stub predictor whose forward is
cheap but shaped exactly like the real ``Qwen3TTSTalkerCodePredictorModel``
(seen in a runtime trace of the stock path):

  prefill:   forward(inputs_embeds=(B,2,H))          -> logits (B,2,V), gs=1
  step k:    forward(input_ids=(B,1), gs=k, cache)   -> logits (B,1,V), gs=k+1

CUDA-graph behavior itself needs a real GPU and is covered by
``scripts/gpu_validate.py`` on hardware; here we verify the loop logic,
token selection, and patch/unpatch mechanics. Skipped when torch is
absent (the suite must pass without the qwen3 extra installed).
"""

import pytest

torch = pytest.importorskip("torch")


class StubPredictor:
    """Minimal stand-in for the code predictor: logits put all mass on
    token id == the predictor's returned generation_steps, so the greedy
    sequence counts 1, 2, 3, ... exactly as the position threading dictates."""

    def __init__(self, vocab: int = 2048):
        self.vocab = vocab
        self.calls: list[dict] = []

    def generate(self, **kw):  # stock path stand-in (class-level, like GenerationMixin)
        raise AssertionError("stock generate must not be called in fast-path tests")

    def forward(self, input_ids=None, inputs_embeds=None, generation_steps=None,
                past_key_values=None, use_cache=None, **kw):
        from types import SimpleNamespace

        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            gs_in = None
            gs_out = inputs_embeds.shape[1] - 2 + 1
        else:
            gs_in = generation_steps
            gs_out = gs_in + 1
        self.calls.append({"inputs_embeds": inputs_embeds, "input_ids": input_ids,
                           "gs_in": gs_in})
        n = inputs_embeds.shape[1] if inputs_embeds is not None else 1
        logits = torch.full((1, n, self.vocab), -10.0)
        logits[0, n - 1, gs_out] = 10.0  # last position predicts its own gs_out
        return SimpleNamespace(logits=logits, past_key_values=object(),
                               generation_steps=gs_out)


def test_fast_generate_greedy_matches_stock_contract():
    from tts_server.backends.qwen3_fast import make_fast_generate

    fast = make_fast_generate(do_sample=False, temperature=1.0, top_k=0, top_p=1.0)
    p = StubPredictor()
    out = fast(p, inputs_embeds=torch.zeros(1, 2, 8))
    # 1 prefill + 14 single-token forwards, fixed-length (1, 15) sequences
    assert len(p.calls) == 15
    assert out.sequences.shape == (1, 15)
    assert p.calls[0]["inputs_embeds"] is not None
    assert all(c["input_ids"] is not None for c in p.calls[1:])
    # greedy over the stub logits walks 1,2,3,...
    assert out.sequences[0, :3].tolist() == [1, 2, 3]
    # generation_steps threading: prefill(None->1), then 1..14
    assert [c["gs_in"] for c in p.calls] == [None] + list(range(1, 15))


def test_fast_generate_sampling_shapes():
    from tts_server.backends.qwen3_fast import make_fast_generate

    torch.manual_seed(0)
    fast = make_fast_generate(do_sample=True, temperature=0.9, top_k=50, top_p=1.0)
    p = StubPredictor()
    out = fast(p, inputs_embeds=torch.zeros(1, 2, 8))
    assert out.sequences.shape == (1, 15)
    assert out.sequences.dtype == torch.int64


def test_patch_and_unpatch_roundtrip():
    from tts_server.backends.qwen3_fast import patch_code_predictor, unpatch_code_predictor

    p = StubPredictor()
    stock = p.generate
    patch_code_predictor(p, greedy=True)
    assert "generate" in p.__dict__ and p.generate is not stock
    unpatch_code_predictor(p)
    assert "generate" not in p.__dict__
    # after unpatch, attribute lookup falls back to the class (stock path)


def test_greedy_select_is_argmax():
    from tts_server.backends.qwen3_fast import _select

    logits = torch.tensor([[1.0, 3.0, 2.0]])
    tok = _select(logits, do_sample=False, temperature=1.0, top_k=0, top_p=1.0)
    assert tok.shape == (1, 1)
    assert tok.item() == 1


def test_top_k_filters_low_probability_tokens():
    from tts_server.backends.qwen3_fast import _select

    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, -50.0]])
    # with top_k=2 the -50 token is masked; 1000 draws never pick it
    picks = torch.cat([
        _select(logits, do_sample=True, temperature=1.0, top_k=2, top_p=1.0)
        for _ in range(200)
    ])
    assert picks.unique().tolist() in ([0], [1], [0, 1])
