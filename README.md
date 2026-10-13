# FuZev EXL3 Quantization Integration

EXL3 quantization for vLLM, installed as `vllm-exl3`. It registers
`--quantization exl3` and runs packed routed experts, dense EXL3 linears and
row-wise n-gram embedding tables through ExLlamaV3's CUDA kernels. This
optional component is **AGPL-3.0-only**.

Part of [FuZev](https://github.com/PeisongXiao/fuzev):

- **Engine:** the [vllm-fuzev](https://github.com/PeisongXiao/vllm-fuzev) fork,
  with ExLlamaV3's CUDA extension from [exllamav3-fuzev](https://github.com/PeisongXiao/exllamav3-fuzev).
- **Decisions:** [fuzev-vllm-plugin](https://github.com/PeisongXiao/fuzev-vllm-plugin)
  with the [published head](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev).
- **Results:** vLLM EXL3 results are in the
  [FuZev benchmark results](https://github.com/PeisongXiao/fuzev/blob/main/docs/benchmarks.md).

**Contents**

- [Credits and provenance](#credits-and-provenance)
- [Installation](#installation)
- [Qwen Serving View](#qwen-serving-view)
- [Limitations](#limitations)
- [License](#license)
- [Upstream vllm-exl3](#upstream-vllm-exl3)

## Credits and provenance

Please credit **vcruz305** and the upstream work this project builds on:

- **Turboderp / [ExLlamaV3](https://github.com/turboderp-org/exllamav3)**: EXL3 trellis format, MCG/mul1 codebooks, quantization math, packed execution, and reused extension kernels/headers.
- **Mia's AI Lab / [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)**, including @plotarmordev: substantial routed-expert integration lineage and the historical MIT-licensed E2 fat-GEMM sources.
- **vLLM**: integration and checkpoint-loading interfaces identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Exact copied/derived files, historical notices, and the distinction between adapted design and independent implementation are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/provenance.md](docs/provenance.md).

FuZev fork maintainer: **Peisong Xiao**.

## Installation

Version `0.5.0+fuzev.v1.0`, based on upstream vllm-exl3 v0.5.0 plus its
unreleased per-layer MoE TP chunk rotation (#40; [CHANGELOG](CHANGELOG.md#unreleased)).

Install both engine forks, the runtime and the plugins into one environment
with the canonical [FuZev install procedure](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#vllm-with-exl3).
The default quantization build uses ExLlamaV3's CUDA extension for packed
execution. For `vllm_exl3_c`, follow the [optional native build](docs/fuzev-serving.md#optional-native-build).

## Qwen Serving View

From this repository's source directory, create a complete index and derived
configuration for the local checkpoint:

```bash
python tools/exl3_pack_tools/qwen_pack_view.py /absolute/path/to/exl3-snapshot ./qwen-exl3-view
```

- [Serving-view guide](docs/fuzev-serving.md#create-a-serving-view): the model integration.
- [Disk-backed packed tables](docs/fuzev-serving.md#disk-backed-packed-tables):
  optional, with a complete PIECEWISE graph configuration.
- [Qwen single-Spark recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe):
  upstream vLLM setup.

## Limitations

Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).
See [serving limitations](docs/fuzev-serving.md#limitations).

## License

- This project is **AGPL-3.0-only** ([LICENSE](LICENSE)).
- Apache-2.0 material from earlier vllm-exl3 releases and from vLLM keeps its
  license in [LICENSE.APACHE-2.0](LICENSE.APACHE-2.0).
- Third-party material retains its notices in [NOTICE](NOTICE) and
  [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).
- Model weights and runtime dependencies have separate licenses.

## Upstream vllm-exl3

The sections below are upstream vllm-exl3's README. They describe upstream's
recipes, test protocol and experimental features; FuZev's measured configuration
is in the [serving guide](docs/fuzev-serving.md).

![vllm-exl3 — EXL3 quantization plugin for routed MoE serving](assets/header.png)

[![Follow on X](https://img.shields.io/badge/Follow-%40ViC305-black?logo=x)](https://x.com/ViC305) [![Follow on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Follow-vcruz305-yellow)](https://huggingface.co/vcruz305)

### Upstream Development Status

| Change | What is implemented | Qualification boundary |
|---|---|---|
| Per-bit native row caps | K2/K3/K4 overrides wrap the real native resolver during `register()` | Opt-in experiments; defaults and unsupported-shape guards remain |
| Gate/up SUH compatibility cache | Normal fused-state construction caches whether the input rotations match; direct fat-path callers cache on first use | Rotations must remain immutable after construction; new speedup not yet measured |
| Runtime diagnostics | Backend preference, extension ABI, row caps, registration state, scratch and grouped-planner status | Describes the calling process, not a remote serving worker |
| K2/K3 grouped-prefill planner | Default-off eligibility checks and bounded candidate row-window estimates | **No grouped CUDA executor; `execution_available` is false** |
| Fused scratch request | Reports the requested row capacity separately from the actual capacity | **Override inactive; actual `TEMP_ROWS_FUSED` remains 2048** |
| ExLlamaV3 CPU-offload planner | Fail-closed metadata/architecture eligibility plan for an external ExLlamaV3 CPU-MoE experiment | **No CPU-compute executor is implemented in `vllm-exl3`** |
| **vLLM UVA expert guard** | Requires the six large packed expert payloads to be vLLM UVA-mapped before post-load EXL3 handles/pointers are accepted | **Experimental placement gate only; real-GPU parity/performance not yet qualified** |

Neither enabling the grouped planner nor requesting fewer fused rows currently changes serving allocations or provides a kernel speedup. Planner scratch estimates are not a measurement or bound on every existing runtime allocation.

Release history and earlier kernel work belong in [CHANGELOG.md](CHANGELOG.md). Historical K4 fat-GEMM microbenchmarks are not evidence of K2/K3 grouped-prefill acceleration.

### Mixed-K fused dispatch

Mixed-K packs (layers with heterogeneous per-expert bit widths) use
`exl3_moe_mixedk` when the installed ExLlamaV3 extension provides that CUDA
kernel. Otherwise the plugin uses the eager per-expert reference loop.

## Compatibility and execution

Existing integrations include `Glm5Next`, `DeepseekV4`, DeepSeek-V4.1 compatibility helpers for a V4.1-capable vLLM runtime, and `Qwen4ExpForConditionalGeneration`, each requiring its matching model plumbing. The plugin does not make an unsupported model architecture appear in either vLLM or standalone ExLlamaV3.

The upstream release this integration surface is checked against is **vLLM v0.30.0**. Per-point state — what resolves, what moved, and the one integration that does not carry over — is in [docs/VLLM_COMPATIBILITY.md](docs/VLLM_COMPATIBILITY.md). `tools/check_vllm_compat.py <vllm-tree>` re-runs that check against any tree or release tag.

Start with the [GLM single-Spark recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) and its [TP1 qualification protocol](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe/blob/main/docs/TP1_POLICY_AB.md).

### DeepSeek-V4.1-Flash

DeepSeek-V4.1 support has a strict ownership boundary:

- a V4.1-capable **vLLM** runtime owns `DeepseekV41ForCausalLM`, CED/CSA2 attention, Engram, routing and DSpark;
- `vllm-exl3` bridges EXL3 metadata and routed-expert execution into those vLLM layers;
- this plugin does **not** implement a standalone V4.1 model class;
- current upstream ExLlamaV3 does not provide the forward-correct V4.1/CED/Engram model graph required to load the checkpoint standalone.

The public compiled checkpoint under active qualification is [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw). Use the [DeepSeek-V4.1 recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe), which now contains TP2/TP4 Spark paths plus a separate one-GPU `sm_120`/large-host-RAM UVA qualification path.

#### Experimental vLLM UVA expert placement

Current vLLM exposes selective UVA weight offload and a DeepSeek-V4.1 Engram CPU/UVA mode. That makes a one-small-GPU/large-host-RAM experiment possible **without** first porting the full V4.1 graph into standalone ExLlamaV3.

When the recipe sets:

```bash
export VLLM_EXL3_REQUIRE_UVA_EXPERTS=1
```

this plugin requires all six large packed expert payload segments to carry vLLM's mapped-UVA placement marker before it builds the normal EXL3 handles/pointer tables:

```text
w13_trellis  w13_suh  w13_svh
w2_trellis   w2_suh   w2_svh
```

The expected vLLM CLI shape is:

```bash
--offload-backend uva \
--cpu-offload-gb <budget> \
--cpu-offload-params \
  w13_trellis w13_suh w13_svh \
  w2_trellis w2_suh w2_svh
```

with V4.1 Engram separately requested using:

```bash
--engram-config '{"cpu_offload":true}'
```

This is **not CPU compute**. The packed expert weights live in pinned host memory as mapped accelerator views and the GPU EXL3 kernels dereference them over UVA/PCIe. The guard rejects ordinary-CPU fallback placement, partial placement, and a requested run where none of the six payloads are actually UVA-mapped.

A successful guard is only a placement gate. It does not prove output parity, acceptable PCIe throughput, CUDA-graph behavior or that the host can pin hundreds of GiB. See [`docs/CPU_OFFLOAD.md`](docs/CPU_OFFLOAD.md).

#### External ExLlamaV3 CPU-compute planner

The plugin also exposes a policy-only planner for the separate ExLlamaV3 CPU-MoE route:

```python
from vllm_exl3 import plan_exllamav3_cpu_offload

plan = plan_exllamav3_cpu_offload(
    architecture="DeepseekV41ForCausalLM",
    codebooks="mul1",
    max_k=4,
    uniform_expert_biases=True,
    exllamav3_architecture_available=False,
)
print(plan.to_dict())
```

`plan.vllm_exl3_execution_available` is intentionally `False`. That helper records the current upstream ExLlamaV3 CPU-MoE eligibility contract; it does not enable a hidden plugin CPU backend.

Routed expert weights remain packed at load time. Some fallback/prefill paths reconstruct **temporary FP16 weights for an expert**; packed loading does not mean zero reconstruction or zero scratch memory. The existing tiled fat-GEMM fast path is gated to eligible **K4/MCG**, non-mul1 projections with compatible gate/up input rotations. K2/K3 and distinct-rotation cases retain their applicable fallback paths.

Native MoE ABI 2 includes hidden width 4096 and local intermediate widths 1024/2048 with K2/K3/K4 and optional SwiGLU clipping. These are kernel contract dimensions, not a promise that every model or row count uses the native path. Unsupported cases may fall back. Verify actual dispatch, not only the requested backend or presence of an extension symbol.

## Host-memory offload boundary

There are two distinct host-memory modes:

1. **vLLM UVA zero-copy:** packed EXL3 parameters can be pinned in host RAM and exposed as mapped accelerator views; the GPU remains the expert compute device. The new guard supports qualification of this placement.
2. **ExLlamaV3 CPU-MoE:** experts live/compute on CPU. `vllm-exl3` does not implement this executor. Current upstream ExLlamaV3's experimental CPU-MoE requires `mul1`, K <= 8, uniform expert-bias presence, and an architecture ExLlamaV3 can instantiate.

For DeepSeek-V4.1 on a small GPU, the vLLM UVA route is now the preferred first experiment because it preserves vLLM's existing V4.1 graph. If it is too PCIe-bound or cannot pin enough memory, the standalone ExLlamaV3 V4.1 + CPU-compute route remains the fallback.

More detail: [`docs/CPU_OFFLOAD.md`](docs/CPU_OFFLOAD.md).

## Pack metadata

For the routed-expert packs used by the GLM recipe, the basic declaration is:

```json
{
  "quantization_config": {
    "quant_method": "exl3",
    "bits": 2,
    "codebook": "mcg"
  }
}
```

Mixed packs additionally declare per-layer overrides in `layer_bits`. Preserve the checkpoint's existing config, index, tensor shapes and bit-width metadata; do not flatten a mixed pack to the base `bits` value.

Other supported metadata includes `non_routed_quantization` for delegating source-format non-routed weights and `non_routed_exl3` for declared dense EXL3 linears. Native ExLlamaV3 packs can also use per-tensor widths, mul1 codebooks, padding and row-wise n-gram embeddings. These capabilities do not make every specialized kernel eligible. See `src/vllm_exl3/exl3.py` and `tools/exl3_pack_tools/` for the serving-side contract.

Never infer a codebook from average bpw or the repository name. Inspect the checkpoint's actual metadata/tensor suffixes first.

## Runtime controls and diagnostics

Set controls **before starting the serving process**, and restart that process between variants.

| Control | Meaning |
|---|---|
| `VLLM_EXL3_MOE_KERNEL=auto\|native\|exllamav3` | Backend preference; shape/format checks and fallback still apply |
| `VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2`, `_K3`, `_K4` | Per-bit native row ceiling; absent/invalid values fall back to the existing resolver, zero disables eligibility for that width |
| `VLLM_EXL3_FAT_THRESHOLD` | Existing fat-expert routing threshold; default 256 |
| `VLLM_EXL3_FUSED_TEMP_ROWS` | Requested capacity only; no active allocation override |
| `VLLM_EXL3_GROUPED_PREFILL` | Requests experimental planner eligibility only; default off |
| `VLLM_EXL3_GROUPED_PREFILL_MAX_ROWS` | Planner row-window budget, not a live kernel allocation setting |
| `VLLM_EXL3_REQUIRE_UVA_EXPERTS=1` | Require complete mapped-UVA placement of the six large EXL3 MoE payload segments before post-load handle construction |
| `VLLM_EXL3_NGRAM_KERNEL=ext\|torch` | n-gram row decoder: the compiled `exllamav3_ext.ngram_dequant` kernel (default) or the pure-torch twin |
| `VLLM_EXL3_NGRAM_TABLE=resident\|disk` | Where the packed n-gram table lives. `resident` (default): one int16 device tensor. `disk`: the checkpoint's memory-mapped views stay the table and each lookup gathers its rows on the host, so the table costs page cache instead of 32 to 36 GiB of device memory; needs `--engram-config '{"cpu_offload":false}'` and `--compilation-config '{"cudagraph_mode": "PIECEWISE", "splitting_ops": [<attention ops>, "vllm::exl3_ngram_lookup_out"]}'` ([complete list](docs/fuzev-serving.md#disk-backed-packed-tables)) because the host gather cannot sit inside a captured graph |

In a **fresh process in the intended runtime environment**, explicitly register before inspecting the policy:

```python
import json
import vllm_exl3

vllm_exl3.register()
print(json.dumps(vllm_exl3.runtime_diagnostics(), indent=2, sort_keys=True))
```

For the UVA experiment, inspect `uva_expert_offload.requested`, `guard_installed`, and `parameter_segments` in addition to the normal native/grouped diagnostics. A standalone diagnostic is a preflight, not proof of what an already-running server loaded. Capture worker-side placement and dispatch evidence for benchmarks.

Speculation/context helpers are callable utilities. Their existence does not establish that a vLLM runner uses an adaptive schedule, or that a generic MLA memory estimate models GLM's hybrid caches, draft state, workspace and graph pools.

## Validation

From the matching source checkout:

```bash
python -m pytest -q tests/test_runtime_policy.py tests/test_prefill_policy.py
python -m pytest -q tests/test_native_moe_contract.py tests/test_fat_distinct_suh.py
python -m pytest -q tests/test_cpu_offload_plan.py tests/test_uva_offload.py
```

GPU tests that skip are **not passes for hardware qualification**. Run the full suite as well and retain the skip/failure report. Native tests, real-checkpoint parity, changed-input/routing graph replay, memory behavior and end-to-end serving are separate gates.

Compare exact revisions on the same checkpoint and workload. Record cold-prefix TTFT separately from warm prefix-cache hits, output-token accounting including reasoning, server-side runtime identity, actual dispatch, acceptance, peak allocated/reserved memory, host memory headroom and exact launch flags. Keep kernel microbenchmarks separate from full-model performance, and aggregate throughput separate from per-request speed.
