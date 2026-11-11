from ai_result import build_ai_fragments, resolve_ai_percent
from ai_scoring import load_ai_config, sigmoid_ai_percent


def test_resolve_null_when_disabled_or_empty(monkeypatch):
    monkeypatch.delenv("AI_DETECT_D0", raising=False)
    monkeypatch.delenv("AI_DETECT_SCALE", raising=False)
    cfg = load_ai_config()
    assert resolve_ai_percent(enabled=False, scores=[1.0], config=cfg) is None
    assert resolve_ai_percent(enabled=True, scores=[], config=cfg) is None


def test_resolve_percent_when_scores_present(monkeypatch):
    monkeypatch.setenv("AI_DETECT_D0", "1.0")
    monkeypatch.setenv("AI_DETECT_SCALE", "1.5")
    cfg = load_ai_config()
    got = resolve_ai_percent(enabled=True, scores=[0.0, 0.0], config=cfg)
    assert got == sigmoid_ai_percent(0.0, 1.0, 1.5)


def test_build_ai_fragments_fields_and_limit(monkeypatch):
    monkeypatch.setenv("AI_DETECT_D0", "1.0")
    monkeypatch.setenv("AI_DETECT_SCALE", "1.5")
    cfg = load_ai_config()
    pairs = [("один" * 100, 0.0), ("два", 2.0), ("три", -1.0)]
    fr = build_ai_fragments(pairs, cfg, max_fragments=2, preview_chars=10)
    assert len(fr) == 2
    assert fr[0]["text"] == ("один" * 100)[:10]
    assert fr[0]["d_score"] == 0.0
    assert fr[0]["ai_probability"] == sigmoid_ai_percent(0.0, 1.0, 1.5)
    assert fr[1]["text"] == "два"


def test_build_ai_fragments_zero_limit_returns_empty(monkeypatch):
    cfg = load_ai_config()
    assert build_ai_fragments([("x", 1.0)], cfg, max_fragments=0) == []
