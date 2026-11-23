---
pretty_name: "Inkling 2 Mini · CIRU Halo Agent (gfx1151)"
base_model: thinkingmachines/Inkling-2-Mini
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - vllm
  - rocm
  - amd
  - strix-halo
  - gfx1151
  - long-context
  - speculative-decoding
  - multimodal
---

# Inkling 2 Mini · CIRU Halo Agent (gfx1151)

Run Thinking Machines Lab's Inkling 2 Mini locally on **AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`)**. With **v1.5.0**, the default `./serve.sh` launch runs a **complete native RDNA 3.5 HIP prefill and decode engine** out of the box—no `hipcc`, no hidden flags, and no vision speed penalty: **2,299.4 client / 2,421.9 server prompt tokens/s** at 4K (**+56% over v1.3.1**, **2,309.1 / 2,368.0 tok/s at 8K**, **2,100.8 / 2,117.8 tok/s at 32K**, **1,904.3 / 1,915.5 tok/s at 58K**, all with zero prefix-cache hits), **50.2 server / 46.8 client decode tokens/s at 128K context** (**+41.1% server / +38.6% client** with 100% exact 128K needle retrieval and `1M`-ready `int64_t` paged KV addressing), **194–208 tok/s** on heavy 8-stream decode (**+13–19% faster**, **246–296 tok/s** on `B=8` exact JSON), and **`65 s` fresh-cache cold startup** (**3.5× faster** than v1.4.0's `230 s+` cold boot). The OpenAI-compatible API accepts text and images, enables prefix caching, streams reasoning, supports native tool calling, defaults to a **1,048,576-token context** on 128 GB Strix Halo, and idles at **~0.5–1.4% of one CPU core**.

![Inkling 2 Mini CIRU Halo Agent benchmark on AMD Ryzen AI Strix Halo](assets/inklingamd.png)

*Measured on AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, 128 GB unified memory) with CIRU v1.5.0 and adaptive DSpark speculative decoding. See the tables and scaling notes below for exact v1.5.0 warm zero-cache prefill (`1K–58K`), `128K` context decode & retrieval, concurrent `B=8` scaling, and full-vocabulary KL quality verification.*

---

## Release notes

### v1.5.0 — All-HIP RDNA 3.5 Prefill & Decode Engine (`+41%` 128K Decode, `2,422` Server PP @ 4K, `65s` Cold Startup)
- **Unified portable ABI v3 native HIP library (`scripts/rocm/hip_inkling_gfx1151_671b45b80972.so`):** Compiles **299 native RDNA 3.5 (`gfx1151`) GPU kernels** across all 7 `.hip` translation units with **0 VGPR/SGPR/Scratch spills**, zero `RPATH`/`RUNPATH`, strictly `libamdhip64.so.7` + `libc.so.6` (`GLIBC_2.2.5`), verified by `MANIFEST.sha256`—requiring no `hipcc` or `rocm-sdk-devel`, with 7 independent per-unit kill switches and automatic fallback preserved.
- **`1M`-ready native HIP decode attention (`hip_attn_decode.hip`):** Zero-materialization cooperative `BN=32` / `BQ_PACK=32` BF16-WMMA split-KV flash relative attention + `hip_kv_conv_decode` + `hip_decode_combine` covering both Target (`HAS_REL=True`, `HAS_TAU` post-8K) and Draft (`block_decode_attention`) layers with `int64_t` paged KV addressing up to `1,048,576` tokens—sustaining **`227.7 GB/s`** at `128K` global attention and delivering **`50.18` server / `46.79` client decode tok/s at `128K`** (**+41.1% server / +38.6% client**) with **100% exact `128K` Needle-in-a-Haystack retrieval** (`1,482.3 tok/s` prefill).
- **Single-pass native HIP W6/W4 quantized decode (`hip_quant_decode.hip`):** Pipelined `N_TILE=64` single-pass `M_TILE={16,32,48}` WMMA kernels for `w6_qkv`, `w6_wo`, `w6_head` (fused BF16/FP32 softcap), `w4_wmma` (optional bias), `w6_head_gemv_m1`, and exact-tie-break `w4_markov_argmax` (`248.0 GB/s`)—cutting `B=8` `M=40` W6/W4 linear step time from `9.85 ms → 6.32 ms` (**`-35.8%`**).
- **Single-pass-per-expert `M16`/`M32` native HIP decode MoE (`hip_moe_decode.hip`):** Greedy adjacent-block `M32` pairing planner (with capture-safe `w13 → w2` deduplication) + zero-`smem_B` software-pipelined WMMA kernels across all four expert banks (`source`, `mixed1`, `mixed2`, `coda`) plus `hip_moe_reduce` lowered to `M >= 1`—saturating `~220–225 GB/s` (`92–94%` of LPDDR5X peak) and driving **`+13.0%`** (`8×256t`, `194.1 tok/s`) and **`+17.4%–18.7%`** (`8×512t`, `205.5–207.8 tok/s`) live `B=8` throughput gains.
- **Complete native HIP prefill & decode glue + DSpark (`hip_attn_prefill.hip`, `hip_glue_prefill.hip`, `hip_glue_decode.hip`):** Native HIP zero-mat BF16-WMMA prefill attention (`+29%` Unit B prefill), `hip_sconv_prefill`/`hip_sconv_decode` (1-channel/thread 21-VGPR `int32`/`int64`-position safe), 2-pass 1-read-per-weight `hip_loop_decode_blend_rms`, `uint4` `hip_loop_decode_down` & `hip_router_gemm_m8`, 256-thread parallel IEEE-754 bit-exact `hip_route_m8`, `hip_attn_res_prefill` (1-pass register cache up to `N<=12`, lowered to `M>=1`), and full DSpark draft kernels (`mix_rmsnorm`, `rms_norm`, `qk_norm_rope`, `dspark_w8`, `silu_and_mul`, `add_softcap`, `markov_gate`).
- **3.5× faster fresh-cache cold startup:** Eliminates ported prefill and decode Triton/Gluon JIT compilation on the hot path, cutting empty-cache `serve.sh` cold startup to **`65 s`** (`31.3 s` vLLM engine init, **`1 s`** HIP graph capture, down from `230 s+` in v1.4.0).
- **Full-vocabulary quality verified against v1.4.0:** `mean_forward_kl = 0.000192` and **`100.00%` top-1 agreement** (`162/162`) across aligned decode (including 64-step post-8K `tau != None` `9,600t` decode at `0.000043` KL), `0.000616` KL / `99.50%` top-1 on prefill (`200` positions). **Model and fitted weight files (`WEIGHTS.sha256`) are unchanged from v1.2.x–v1.4.0.** [v1.5.0 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.5.0)

### v1.4.0 — Native RDNA 3.5 HIP Prefill Kernels (2,230 Client / 2,363 Server tok/s at 4K, +52% Faster Prefill)
- Added initial native `gfx1151` HIP paired-`M128`/`M64` MoE prefill (`hip_moe_prefill.hip`), 1-warp K/V short-conv, zero-mat loop blend, scalar-page `BLOCK_N=16` prefill attention, and fused BF16-WMMA prefill router. [v1.4.0 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.4.0)

### v1.3.1 — Fast Adaptive Vision Preset as Default (Zero Vision Penalty)
- Moved `--mode vision` onto the fast adaptive stack (`K=7` at `B=1`, `K=4` at `B=2..8`, 9 Target + 9 Draft FULL HIP graphs) and made it the default `./serve.sh` launch. [v1.3.1 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.3.1)

### v1.3.0 — Multi-Agent HIP Graphs, Adaptive DSpark, and Smooth 64K Prefill
- Captured FULL Target + Draft HIP graphs across `1..8` active sequences, dual-depth adaptive DSpark speculation (`K=7 @ B=1`, `K=4 @ B=2..8`), hybrid 64K global prefill attention, and batched `M <= 64` W6/W4/WMMA concurrency kernels (**3.5×** concurrent throughput vs v1.2.2). [v1.3.0 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.3.0)

### Previous releases
- **v1.2.2:** Bundled AMD's ROCr idle polling fix (`libhsa-runtime64.so.1`) for the pinned ROCm 10.0 SDK, lowering idle CPU usage at 1M context to ~1.4% of one core. [v1.2.2 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.2.2)
- **v1.2.1:** Enabled HIP blocking waits before ROCm initialization to prevent idle CPU pinning. [v1.2.1 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.2.1)
- **v1.2.0:** Defaulted text and vision launchers to a configurable **1,048,576-token** context window on 128 GB Strix Halo. [v1.2.0 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.2.0)
- **v1.1.0:** Added the native ROCm `gfx1151` vision encoder (`model.visual.*`) and `--mode vision` preset.
- **v1.0.0:** Initial Inkling 2 Mini CIRU Halo Agent release with RDNA 3.5 FP16 WMMA kernels, fitted expert weights, DSpark speculative decoding, and single-stream HIP graphs.

---

## Performance on AMD Ryzen AI Strix Halo (`gfx1151`)

Measured live end-to-end on an AMD Ryzen AI Max+ 395 with Radeon 8060S graphics (`20 WGPs / 40 CUs`) and 128 GB unified LPDDR5X memory (NixOS, Python 3.12.12, PyTorch `2.13.0+rocm10.0.0`, Triton `3.8.0`, vLLM `0.34.0+rocm100`) running the **default full-featured Vision profile** (`--mode vision --tools --context 1048576 --kv-mib 53248 --batch-tokens 4096`, adaptive `K=7/K=4`, 9+9 FULL HIP graphs, prefix caching enabled, zero `hipcc` on `PATH`).

### 1. Warm Zero-Cache Prompt Processing (`1K–58K`, Default Vision Profile, `Hits = 0`)

Every row used a freshly salted prompt with verified **`0` prefix-cache hits**, reporting **both** client-timed TTFT throughput (`prompt_tokens / TTFT`) and server-timed prefill throughput (`vllm:request_prefill_time_seconds_sum`).

| Prompt tokens | v1.3.1 Server PP | v1.4.0 Default (`bt=4096`) Client / Server PP | **v1.5.0 Default (`bt=4096`) Client / Server PP** | **v1.5.0 Client TTFT** | Gain vs. v1.4.0 (Server PP) |
|---:|---:|---:|---:|---:|---:|
| **945** *(~1K)* | ~1,350 tok/s | 1,732.3 / 1,942.6 tok/s | **1,781.8 / 2,073.0 tok/s** | 0.53 s | **+6.7%** |
| **1,900** *(~2K)* | ~1,480 tok/s | 2,089.6 / 2,242.9 tok/s | **2,159.0 / 2,332.6 tok/s** | 0.88 s | **+4.0%** |
| **3,796** *(~4K)* | ~1,550 tok/s | 2,229.9 / 2,363.0 tok/s | **2,299.4 / 2,421.9 tok/s** | **1.65 s** | **+2.5%** *(+56% vs. v1.3.1)* |
| **7,660** *(~8K)* | ~1,490 tok/s | 2,141.5 / 2,199.6 tok/s | **2,309.1 / 2,368.0 tok/s** | 3.32 s | **+7.7%** |
| **15,345** *(~16K)* | ~1,440 tok/s | 2,049.1 / 2,075.1 tok/s | **2,241.9 / 2,273.3 tok/s** | 6.85 s | **+9.6%** |
| **30,722** *(~32K)* | ~1,360 tok/s | 1,926.0 / 1,944.9 tok/s | **2,100.8 / 2,117.8 tok/s** | **14.62 s** | **+8.9%** |
| **55,187** *(~58K)* | ~1,240 tok/s | 1,752.0 / 1,762.3 tok/s | **1,904.3 / 1,915.5 tok/s** | **28.98 s** | **+8.7%** |

### 2. Long-Context `128K` Decode, Multi-Stream `B=8` Decode, and Cold Startup (Paired Live Gates)

| Workload (Default Vision Profile, `1M / 52 GiB / bt=4096`) | Metric | Triton/Gluon Decode Baseline | **v1.5.0 (All-HIP Decode)** | Delta |
|---|---|---:|---:|---:|
| **`128K` Context Decode (`127,680` cache hits, `128t`)** | **Server tok/s** | 35.56 tok/s | **50.18 tok/s** | **+41.1%** |
| **`128K` Context Decode (`127,680` cache hits, `128t`)** | **Client decode / wall tok/s** | 33.77 / 29.50 tok/s | **46.79 / 38.65 tok/s** | **+38.6% / +31.0%** |
| **`128K` Needle-in-a-Haystack (`pt=127,735`)** | **Exact passkey / Server PP** | Exact / 1,468.8 tok/s | **Exact (100%) / 1,482.3 tok/s** | **+0.9%** |
| **`B=8` Code Generation (`8×512t`, `T=0`)** | **Aggregate Client wall tok/s** | 175.0 tok/s *(21.9/stream)* | **205.5–207.8 tok/s** *(25.7–26.0/stream)* | **+17.4%–18.7%** |
| **`B=8` Decode (`8×256t`, `T=0`)** | **Aggregate Client wall tok/s** | 171.8 tok/s *(21.5/stream)* | **194.1 tok/s** *(24.3/stream)* | **+13.0%** |
| **`B=8` Structured JSON (`8/8` exact)** | **Aggregate Client wall tok/s** | 242.8 tok/s *(v1.4.0)* | **246.1–296.0 tok/s** | **8/8 exact** |
| **`B=1` Decode (`256t` / `512t`, reasoning high)** | **Server / Client wall tok/s** | 69.9 / 68.0 (`256t`), 62.5 / 61.3 (`512t`) | **69.7–70.0 / 68.1**, **62.6 / 61.5–61.7** | Parity |
| **`B=8` `M=40` Complete Step GPU Kernel Sum** | **Median GPU sum (trace)** | 97.18 ms | **89.02 ms** | **-8.16 ms (+9.2%)** |
| **Fresh-Cache Cold Startup (`serve.sh`, zero `hipcc`)** | **Wall / vLLM `init engine`** | 98 s / 62.1 s *(230 s+ in v1.4.0)* | **65 s / 31.3 s** *(1 s graph capture)* | **-33 s (-3.5× vs. v1.4.0)** |

### 3. Functional, Vision, and Full-Vocabulary Quality Verification
- **Full-vocabulary KL audit vs. v1.4.0 (`score_pair`, 200,058-token vocabulary):**
  - **Prefill (`200` sampled positions, `160–9,600t`):** `mean_forward_kl = 0.000616`, `top-1 = 99.50%` (`199/200`).
  - **Decode (`162` aligned verified positions, including 64-step short & 64-step post-8K `tau != None` `9,600t` decode):** **`mean_forward_kl = 0.000192`**, **`top-1 = 100.00%` (`162/162`)** (`post8k_tau_9600_dec64`: `0.000043` KL, `64/64` exact tokens).
- **Live functional & multimodal suite (100% pass):** Math (`17*23=391`), native tool calling (`get_weather({"city":"Tokyo"})`), `B=8` exact JSON (`8/8`), `56t` & `>256t` PNG vision understanding, prefix-cached continuation (`1,216` hits), 4-way ragged mixed batch (`4/4`), `32K` needle (`16.79 s`), `128K` needle (`100%` exact), and `T=0.7` stochastic sampling.

### 4. Original Thinking Machines Lab checkpoint (`--original-weights`)

The same runtime serves the unmodified Thinking Machines Lab checkpoint mirrored in this repository without fitted weight banks. With `--original-weights`, v1.5.0's native HIP `source` NVFP4 prefill & decode MoE, native HIP attention, K/V short-conv, loop controller, and quantized W6/W4 kernels accelerate both prefill and decode automatically across all 48 layers.

---

## Install, upgrade, and run

Install or upgrade the ROCm vLLM runtime from the companion GitHub repository. The installation guide covers GPU access, pinned ROCm 10 wheels, the checked-in installer, upgrade steps from v1.2.x–v1.4.0, launch commands for every mode (`vision`, `concurrent`, `single`, `long-context`, `sampling`), and smoke-test verification. [GitHub runtime](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151) · [Installation guide](INSTALL.md) · [v1.5.0 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.5.0)

This Hugging Face repository contains the unmodified upstream model and DSpark draft checkpoint, the separate CIRU fitted weight banks under `optimized-weights/`, and the full v1.5.0 runtime bundle (including the prebuilt portable `gfx1151` ABI v3 all-HIP library). Download the release with the Hugging Face CLI:

```bash
hf download jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151 --local-dir ./weights
```

**Upgrading from v1.4.0 (or v1.2.x/v1.3.x):** Model and fitted weight files (`WEIGHTS.sha256`) are unchanged. Fetch tag `v1.5.0` in the GitHub runtime repository, install into a new prefix (`~/.local/opt/inkling-halo-v1.5.0`), and restart the server pointing at your existing `--model` directory.

Quick launch examples after installing:

```bash
# Default all-in-one server (vision + prefix cache + 8 slots + 1M ctx + all-HIP prefill & decode + adaptive K=7@B=1 / K=4@B=2..8)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --tools

# Text-only adaptive multi-agent server (same fast stack without vision encoder)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode concurrent --tools

# Dedicated single-stream server (1 slot, K=7, FULL HIP graph)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode single --tools

# Long-context preset (explicit 128K/64K context with large BF16 KV pool)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode long-context --context 131072 --kv-mib 53248 --tools

# Explicit probabilistic K=4 draft proposals (optional for high-temperature stochastic drafting)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode sampling --seqs 8 --vision --prefix-cache --tools
```

Use `--context` to set a different request limit and `--kv-mib` to set the KV cache budget. Context covers prompt, image, and generated tokens together, and active requests share one KV cache.

---

## Credits and license

- Thinking Machines Lab developed Inkling 2 Mini and supplied the original checkpoint and DSpark draft weights. [Upstream model](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)
- CIRU developed the Strix Halo runtime integration, native RDNA 3.5 HIP and Triton kernels, adaptive speculative HIP graphs, and optimized expert weights. The runtime builds on vLLM's Inkling support, Hugging Face Transformers, Triton, PyTorch, and AMD ROCm. [vLLM](https://github.com/vllm-project/vllm) · [Transformers](https://github.com/huggingface/transformers) · [Triton](https://github.com/triton-lang/triton) · [PyTorch](https://pytorch.org/) · [AMD ROCm](https://github.com/ROCm)
- The model weights and CIRU runtime are provided under Apache 2.0. [License](LICENSE) · [Notice](NOTICE)
