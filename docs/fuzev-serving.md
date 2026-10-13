# EXL3 Serving With FuZev

Optional AGPL-3.0-only quantization integration for
[FuZev](https://github.com/PeisongXiao/fuzev). `vllm-exl3` loads packed EXL3
weights into vLLM. The FuZev integration uses:

- `vllm-fuzev` for the model and feature-readout hooks
- `exllamav3-fuzev` for packed CUDA execution
- `fuzev-vllm-plugin` with `fuzev-runtime` for decisions

## Requirements

- **Plugin:** Python 3.11+.
- **Serving:** a CUDA GPU and compatible PyTorch, vLLM and ExLlamaV3 binaries.
- **Sizing:** size the model, KV cache and scratch space for your hardware with
  the engine's settings.

Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130). Recorded vLLM EXL3
installation settings are in the
[FuZev install guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#verified-source-installations).

## Install

Version `0.5.0+fuzev.v1.0`, based on upstream vllm-exl3 v0.5.0 plus its
unreleased per-layer MoE TP chunk rotation (#40; CHANGELOG.md, Unreleased).
The distribution is `vllm-exl3`; the import namespace is `vllm_exl3`.

1. Create one environment with the FuZev install guide's
   [Create An Environment](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#create-an-environment) step.
2. [Install both engine forks](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#install-the-engine):
   vLLM, then ExLlamaV3.
3. Install the [vLLM EXL3 integration](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#vllm-with-exl3).
4. Keep the environment active for serving.

The default Python-only plugin build runs packed operations through
ExLlamaV3's CUDA kernels. When other installed plugins also register `exl3`,
`VLLM_PLUGINS=vllm_exl3` selects this one.

Inspect installed package metadata:

```bash
python - <<'PY'
from importlib.metadata import distribution

package = distribution("vllm-exl3")
print(package.metadata["Name"], package.version)
for entry in package.entry_points:
    if entry.group == "vllm.general_plugins":
        print(entry.name, entry.value)
PY
```

The discovery target is `vllm_exl3.bf16_madv_compat:register`. It registers the
quantization method and the mixed-BF16 source-page reclaim adapter.

### Optional Native Build

The native extension requires:

- PyTorch in the build environment
- A CUDA toolkit with cuSPARSE headers
- A C++ compiler, setuptools >=77 and wheel

Set `EXL3_EXT_INCLUDE` to the `exllamav3_ext` directory of the ExLlamaV3 you
run; it is the builder's only external EXL3 header path. In the install guide's
shell, where `FUZEV_SOURCES` names the extracted sources:

```bash
python -m pip install 'setuptools>=77' wheel
EXL3_EXT_INCLUDE="$FUZEV_SOURCES/exllamav3-fuzev/exllamav3/exllamav3_ext" \
python -m pip install --no-build-isolation "$FUZEV_SOURCES/vllm-exl3-plugin"
```

A requested native build stops with an error when a prerequisite is missing
or compilation fails.

## Create A Serving View

Download the EXL3 checkpoint separately, for example
[turboderp's Qwen3.8-Flash-Next EXL3 pack](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3).
Set `SOURCE` to its local snapshot directory and `VIEW` to a new directory:

```bash
SOURCE=/path/to/exl3-snapshot
VIEW="$PWD/qwen-exl3-view"
python "$FUZEV_SOURCES/vllm-exl3-plugin/tools/exl3_pack_tools/qwen_pack_view.py" "$SOURCE" "$VIEW"
```

The view:

- Links the tensor and tokenizer files. The source files stay unchanged and may
  be read-only.
- Derives `config.json` and `quantization_config.json`.
- Writes a regular `model.safetensors.index.json` from every safetensors header,
  including vision and n-gram embedding files.

The tool prints the view path and tensor-file count. If a source header or
configuration is unusable, it removes the incomplete view; correct the input
and retry.

`vllm-fuzev` builds in the packed LM-head binding and split vision-QKV mapping.
Do not run `patch_vllm_qwen4_ple.py` or `patch_vllm_vision_split.py` on that fork.

## Configure vLLM

For normal generation, run this from the environment above. Context, sequence
count and memory fraction are example settings from the recorded GB10 run. On
unified-memory GPUs, leave memory for the OS.

```bash
export VLLM_PLUGINS=vllm_exl3
vllm serve "$VIEW" --quantization exl3 --served-model-name qwen-exl3 \
  --gpu-memory-utilization 0.75 --max-model-len 8192 --max-num-seqs 4 \
  --engram-config '{"cpu_offload":false}' \
  --host 127.0.0.1 --port 8000 --reasoning-parser qwen3
```

Set `--api-key` before exposing the service beyond localhost.

Packed n-gram embedding tables:

- Require `vllm-fuzev`'s per-layer n-gram embedding (PLE) factory hook.
- Take their storage format from the derived header metadata. The plugin
  ignores the model's `ple_embedding_dtype` hint for them. Unclaimed tables use
  the engine's normal format selection.
- Need raw embedding offload disabled: set `engram_config.cpu_offload` to
  `false`. The plugin manages their storage, resident by default or disk-backed
  below.

### Disk-Backed Packed Tables

For disk-backed packed tables, set `VLLM_EXL3_NGRAM_TABLE=disk`; the engine
rejects `ple_storage: mmap` for them. Keep `engram_config: {"cpu_offload": false}`.

```bash
export VLLM_PLUGINS=vllm_exl3
export VLLM_EXL3_NGRAM_TABLE=disk
vllm serve "$VIEW" --quantization exl3 --served-model-name qwen-exl3 \
  --gpu-memory-utilization 0.75 --max-model-len 8192 --max-num-seqs 4 \
  --engram-config '{"cpu_offload":false}' \
  --load-format safetensors --safetensors-load-strategy lazy \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","splitting_ops":["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen4_exp_ple_short_conv","vllm::qwen4_exp_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::gdn_attention_core_xpu","vllm::olmo_hybrid_gdn_full_forward","vllm::sparse_attn_indexer","vllm::rocm_aiter_sparse_attn_indexer","vllm::deepseek_v4_attention","vllm::hpc_rope_norm_forward","vllm::unified_kv_cache_update","vllm::unified_mla_kv_cache_update","vllm::exl3_ngram_lookup_out"]}' \
  --host 127.0.0.1 --port 8000 --reasoning-parser qwen3
```

Disk mode needs PIECEWISE CUDA graphs; the plugin refuses FULL graph modes at
load. The splitting list is vLLM 0.30.0's defaults plus
`vllm::exl3_ngram_lookup_out`, which keeps the host gather outside captured
graphs.

For decisions, `fuzev-vllm` from
[fuzev-vllm-plugin](https://github.com/PeisongXiao/fuzev-vllm-plugin) serves
generation plus `/v1/systemone` and `/v1/decisions` on the same engine; this
package supplies the quantization. Save this JSON as `exl3-engine.json`, with
your serving-view path in `model`:

```json
{
  "model": "/absolute/path/to/qwen-exl3-view",
  "quantization": "exl3",
  "engram_config": {"cpu_offload": false},
  "async_scheduling": false,
  "gpu_memory_utilization": 0.75,
  "max_model_len": 8192,
  "max_num_seqs": 4,
  "load_format": "safetensors",
  "safetensors_load_strategy": "lazy",
  "compilation_config": {
    "cudagraph_mode": "PIECEWISE",
    "splitting_ops": [
      "vllm::unified_attention_with_output",
      "vllm::unified_mla_attention_with_output",
      "vllm::mamba_mixer2",
      "vllm::mamba_mixer",
      "vllm::short_conv",
      "vllm::qwen4_exp_ple_short_conv",
      "vllm::qwen4_exp_qsa_with_output",
      "vllm::linear_attention",
      "vllm::qwen_gdn_attention_core",
      "vllm::qwen_gdn_attention_core_fused_norm_packed",
      "vllm::gdn_attention_core_xpu",
      "vllm::olmo_hybrid_gdn_full_forward",
      "vllm::sparse_attn_indexer",
      "vllm::rocm_aiter_sparse_attn_indexer",
      "vllm::deepseek_v4_attention",
      "vllm::hpc_rope_norm_forward",
      "vllm::unified_kv_cache_update",
      "vllm::unified_mla_kv_cache_update",
      "vllm::exl3_ngram_lookup_out"
    ]
  }
}
```

Start it with a [head bundle](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#head-and-model-files)
for the loaded model:

```bash
export VLLM_PLUGINS=vllm_exl3
export VLLM_EXL3_NGRAM_TABLE=disk
export VLLM_USE_V2_MODEL_RUNNER=0
fuzev-vllm --engine-config exl3-engine.json --model-name qwen-exl3 \
  --head-directory "$FUZEV_HEAD"
```

After any of these servers starts, send a generation request:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-exl3","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

- HTTP 200 returns text in `choices[0].message.content`. Exact text depends on
  the model and sampling settings.
- Without `enable_thinking: false`, the model thinks first. The `qwen3`
  reasoning parser returns the thinking separately from `message.content`.
- The [plugin guide](https://github.com/PeisongXiao/fuzev-vllm-plugin/blob/main/README.md#install-and-serve)
  shows a decision request for `fuzev-vllm`.

## Changes Against Upstream

Changes to [upstream vllm-exl3 v0.5.0](https://github.com/vcruz305/vllm-exl3)
plus #40:

| Path | Change |
| --- | --- |
| `src/vllm_exl3/exl3.py` | Packed PLE factory hook; decoded rows cast to the requested output dtype; disk n-gram lookup op uses vllm-fuzev's capture break (plain op on stock vLLM) |
| `setup.py`, `.github/workflows/ci.yml` | Native `vllm_exl3_c` build only when `EXL3_EXT_INCLUDE` is set; Python-only by default |
| `pyproject.toml`, `VERSION`, `MANIFEST.in` | Release version from `VERSION`; Python 3.11+; native sources and pack tools in source distributions |
| `tools/exl3_pack_tools/qwen_pack_view.py` | Read-only source view with derived config and complete local index |
| `tools/exl3_pack_tools/regenerate_safetensors_index.py` | Shared header indexing and atomic index replacement |
| `tests/test_pack_tools_ngram_fixture.py` | Golden indexes, source preservation, header-only reads and retry after invalid input |
| `tests/test_ple_factory.py`, `tests/test_native_pack_unit.py` | PLE selection, packed storage and decoded output, packed QKV byte preservation |
| `tests/test_ngram_layouts.py`, `tests/test_madv_view_range.py` | Packed table layouts, graph-separated disk lookup and source view preservation |
| `tests/test_attribution_and_sanitization.py` (removed) | Source-text and release-string assertions |
| `README.md`, `docs/fuzev-serving.md`, `docs/VLLM_COMPATIBILITY.md`, `tools/exl3_pack_tools/README.md`, `tools/patch_vllm_qwen4_exp/README.md` | Install, serving and pack-tool instructions |
| `AGENTS.md` (removed) | Upstream coding-agent notes |
| `src/vllm_exl3/exl3.py` (#40) | Upstream per-layer MoE TP chunk rotation |
| `NOTICE`, `CITATION.cff` | Copyright for the FuZev changes; upstream credits retained |

Upstream credits and license notices are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md),
[NOTICE](../NOTICE) and [CITATION.cff](../CITATION.cff).

## Limitations

- EXL3 weights with MTP are unsupported on vllm-fuzev in v1.0.
- The [installation record](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#verified-source-installations)
  covers text-API serving of turboderp's full 4.05 bpw checkpoint on GB10.
  Other packs are untested.
- FuZev's vLLM EXL3 benchmark ([results](https://github.com/PeisongXiao/fuzev/blob/main/docs/benchmarks.md))
  has 5,075 requests. It is the FuZev configuration that uses this plugin.
- Kernel selection depends on the checkpoint geometry, codebook and extension ABI.
