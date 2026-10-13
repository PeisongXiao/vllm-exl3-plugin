"""Unclaimed PLE tables use the engine's format selection."""

from vllm_exl3.exl3 import Exl3Config


def test_unclaimed_ple_returns_none():
    assert Exl3Config(bits=4).get_ple_embedding_method("model.ngram_embedding") is None
    spec = {
        "bits": 6, "num_shards": 1, "rows_per_shard": 32,
        "num_heads": 2, "modules": ["other_table"],
    }
    config = Exl3Config(bits=4, ngram_embedding=spec)
    assert config.get_ple_embedding_method("model.ngram_embedding") is None
