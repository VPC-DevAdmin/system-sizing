"""A staged model whose snapshot has no Hugging Face tokenizer files must
not fall through to an online Hub load (which stalls loadgen workers)."""

from simulator import tokenizer_corpus as tc


def test_staged_without_tokenizer_skips_the_hub(tmp_path, monkeypatch):
    snap = tmp_path / "hub" / "models--acme--tekken-only" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "tekken.json").write_text("{}")
    monkeypatch.setattr(tc, "_local_cache_dirs", lambda: [str(tmp_path / "hub")])
    calls = []

    import transformers

    def fake_load(model_id, **kw):
        calls.append(kw.get("local_files_only", False))
        raise OSError("no tokenizer files")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_load)
    tc._load_tokenizer.cache_clear()
    assert tc._load_tokenizer("acme/tekken-only") is None
    assert calls == [True]          # offline only; never online
    tc._load_tokenizer.cache_clear()


def test_unstaged_model_still_tries_online(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "_local_cache_dirs", lambda: [str(tmp_path / "hub")])
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty"))
    calls = []

    import transformers

    def fake_load(model_id, **kw):
        calls.append(kw.get("local_files_only", False))
        raise OSError("nope")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_load)
    tc._load_tokenizer.cache_clear()
    assert tc._load_tokenizer("acme/not-staged") is None
    assert calls == [True, False]
    tc._load_tokenizer.cache_clear()
