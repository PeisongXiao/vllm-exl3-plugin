"""CPU-only unit tests for native-pack support: padding, n-gram row geometry,
config validation, codebook marker checks, opaque-op registration, and the
dense linear weight loader's shard-span handling.
"""

import pytest
torch = pytest.importorskip("torch")

from vllm_exl3 import exl3 as X


def test_pad128():
    assert X._exl3_pad128(4304) == 4352
    assert X._exl3_pad128(2560) == 2560


def test_ngram_words_per_row():
    assert X.ngram_words_per_row(5) == 51
    assert X.ngram_words_per_row(3) == 31


def test_ngram_embedding_config_and_lookup():
    cfg = X.Exl3Config(
        bits=3,
        codebook="mul1",
        ngram_embedding={
            "bits": 5,
            "num_shards": 4,
            "rows_per_shard": 1024,
            "num_heads": 2,
            "modules": ["ngram_embedding"],
        },
    )
    spec = cfg._ngram_embedding_spec("model.layers.1.ple.ple_embedding.ngram_embedding")
    assert spec is not None
    assert spec["bits"] == 5

    assert cfg._ngram_embedding_spec("model.layers.1.ple.kv_proj") is None

    with pytest.raises(ValueError):
        X.Exl3Config(bits=3, codebook="mcg", ngram_embedding={"bits": 0})


def test_modern_ple_factory_keeps_packed_disk_storage_and_decoded_rows(monkeypatch):
    modern = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ngram_embedding")
    monkeypatch.setenv(X.NGRAM_TABLE_ENV, "disk")
    cfg = X.Exl3Config(bits=4, codebook="mul1", ngram_embedding={
        "bits": 6, "num_shards": 1, "rows_per_shard": 32,
        "num_heads": 2, "sharded": False, "modules": ["ngram_embedding"],
    })
    method = modern.Qwen4ExpPLEEmbeddingMethod.from_quant_config(
        cfg, "model.layers.1.ple.ple_embedding.ngram_embedding", "float8_e4m3fn")
    assert method.table_mode == "disk"
    assert not method.requires_device_loading
    rows = torch.tensor([[1., 2.]], dtype=torch.float16)
    assert torch.equal(method.dequantize(None, rows, torch.float16), rows)
    assert method.dequantize(None, rows, torch.bfloat16).dtype == torch.bfloat16


def test_check_moe_codebook_markers():
    n = 4
    zeros = torch.zeros(n, 1, dtype=torch.int32)
    mul1_all = torch.full((n, 1), X.MUL1_MARKER_SIGNED_INT32, dtype=torch.int32)
    mcg_all = torch.full((n, 1), X.MCG_MARKER_SIGNED_INT32, dtype=torch.int32)

    # All mul1: ok.
    X._check_moe_codebook_markers(zeros, mul1_all, "test")
    # All mcg: ok.
    X._check_moe_codebook_markers(mcg_all, zeros, "test")
    # Neither set: raises.
    with pytest.raises(RuntimeError):
        X._check_moe_codebook_markers(zeros, zeros, "test")
    # Both set: raises.
    with pytest.raises(RuntimeError):
        X._check_moe_codebook_markers(mcg_all, mul1_all, "test")


def test_register_opaque_layer():
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.gate_proj"
    name = X._exl3_register_opaque_layer(layer, "linear")
    assert name == f"exl3_linear:{layer.prefix}"


def test_dense_loader_span_split():
    cfg = X.Exl3Config(bits=2, codebook="mcg")
    m = X.Exl3LinearMethod(cfg, bits=2)

    layer = torch.nn.Module()
    layer.tp_rank = 0
    layer.tp_size = 1

    loader = m._make_weight_loader("svh", 3, [16, 16, 32], False, [], layer, False)

    param = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    loader(param, torch.arange(64, dtype=torch.float16), (0, 1, 2))
    assert torch.equal(param.data, torch.arange(64, dtype=torch.float16))

    param2 = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    loader(param2, torch.arange(64, dtype=torch.float16), None)
    assert torch.equal(param2.data, torch.arange(64, dtype=torch.float16))

    bad_param = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    with pytest.raises(RuntimeError):
        loader(bad_param, torch.arange(10, dtype=torch.float16), (0, 1))


def test_packed_qkv_loader_preserves_trellis_scales_and_signed_markers():
    method = X.Exl3LinearMethod(X.Exl3Config(bits=6, codebook="mul1"), bits=6)
    layer = torch.nn.Module()
    layer.tp_rank, layer.tp_size = 0, 1
    shapes = {"trellis": (8, 24, 96), "suh": (3, 128), "svh": (384,), "mul1": (3, 1)}
    for suffix, shape in shapes.items():
        dtype = (torch.int16 if suffix == "trellis" else
                 torch.int32 if suffix == "mul1" else torch.float16)
        param = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        loader = method._make_weight_loader(suffix, 3, [128, 128, 128], False, [], layer, True)
        values = []
        for index, shard in enumerate(("q", "k", "v")):
            value = (torch.full((8, 8, 96), index + 1, dtype=dtype) if suffix == "trellis" else
                     torch.full((128,), index + 1, dtype=dtype) if suffix != "mul1" else
                     torch.tensor(X.MUL1_MARKER_SIGNED_INT32, dtype=dtype))
            loader(param, value, shard)
            values.append(value)
        if suffix == "trellis":
            expected = torch.cat(values, dim=1)
        elif suffix == "svh":
            expected = torch.cat(values)
        else:
            expected = torch.stack(values).reshape(shape)
        assert torch.equal(param, expected)
