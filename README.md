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

Run Thinking Machines Lab's Inkling 2 Mini locally on **AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`)**. With **v1.3.1**, the default `./serve.sh` launch gives you the full fast adaptive stack out of the box—no hidden flags and no vision speed penalty: **1,691.3 prompt tokens/s** at 3.8K, **1,300.3 prompt tokens/s at 64K (+64.9% over v1.0–v1.2.2)**, **~100 decode tokens/s** on short-context single-request serving (`B=1`, `T=0`), and **310.2–316.6 aggregate output tokens/s** across eight concurrent requests (**3.5× faster** than v1.2.2's `88.92 tok/s`) from one adaptive server. The OpenAI-compatible API accepts text and images, enables prefix caching, streams reasoning, supports native tool calling, defaults to a **1,048,576-token context** on 128 GB Strix Halo, and idles at **~0.5–1.4% of one CPU core**.

![Inkling 2 Mini CIRU Halo Agent benchmark on AMD Ryzen AI Strix Halo](assets/inklingamd.png)

*Measured on AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, 128 GB unified memory) with CIRU v1.3.x and adaptive DSpark speculative decoding. See the scaling notes below for expected throughput at long context (`32K–72K+`), multi-agent bursts, and `T > 0` sampling.*

---

## Release notes

### v1.3.1 — Fast Adaptive Vision Preset as Default (Zero Vision Penalty)
- **Fixed `--mode vision` speed penalty:** In v1.3.0, `--mode vision` forced `probabilistic` fixed `K=4` drafting and disabled the fast `B=1` kernels (`fused W4 Markov argmax`, `W8 draft backbone`, `fused M8 loop`, and adaptive `K=7` at `B=1`). Enabling vision has zero runtime cost during text turns, so v1.3.1 moves `--mode vision` onto the **exact same fast adaptive stack** as `--mode concurrent`: **dual-depth adaptive DSpark (`K=7` at `B=1`, `K=4` at `B=2..8`)**, **9 Target + 9 Draft FULL HIP graphs**, fused W4 Markov argmax, W8 draft backbone, and fused M8 loop—while keeping native ROCm vision, prefix caching, 8 slots, and 1M context enabled.
- **Fast adaptive vision preset is now the default:** Running `"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --tools` with no extra mode flags now launches the full fast adaptive 8-sequence vision + prefix-cache stack directly.
- **Lossless `T >= 0` sampling without forcing slow mode:** Default greedy-draft proposals use exact one-hot rejection sampling in vLLM and remain mathematically lossless for both `temperature = 0` and `temperature > 0`. `--mode sampling` remains available as an explicit opt-in if you specifically want full-vocab `probabilistic` (`K=4`) draft proposals.
- **Clearer real-world speed guidance:** Updated benchmark tables and notes below to show exact throughput across context lengths (`4K` → `72K`), batch concurrency (`B=1` vs. `B=5..8`), and sampling temperature / draft acceptance. **Existing v1.3.0 vLLM runtime (`c95dbfa9`) and model weights (`WEIGHTS.sha256`) are unchanged**—only `scripts/serve.sh` and documentation changed. [v1.3.1 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.3.1)

### v1.3.0 — Multi-Agent HIP Graphs, Adaptive DSpark, and Smooth 64K Prefill
- **3.5× concurrent throughput (`88.9 → 310.2–316.6 agg tok/s` at `B=8`):** Captured **FULL Target + Draft HIP graphs** across `1..8` active sequences (`~1–4 s` startup, `0.31–0.47 GiB`), cutting eager host launch idle time from `37%` down to `4.8%` (**95.2% GPU utilization** across `1,642` kernels/step).
- **Dual-depth adaptive DSpark speculation (`K=7` at `B=1`, `K=4` at `B=2..8`):** Captures both draft depths against one shared model and KV cache—delivering **`100.2 tok/s` for a single active user** and automatically switching to **`K=4` (`310.2 agg tok/s`, `38.8 tok/s/req`) under load**.
- **64K prefill cliff eliminated (`788.5 → 1,300.3 tok/s`, `+64.9%`, 33.9 s faster TTFT):** Added hybrid global prefill attention routing far-history tokens (`>= 2,048` far keys, zero relative bias) through native ROCm AOTriton FlashAttention-2 with an online-softmax near-window Triton merge (`6.46×` faster per global layer at 64K, `0.0000` BF16 max error), plus tuned `KVH=8` split-KV decode scheduling (`238.1 GB/s` at `B=1`, `385 GB/s` effective at `B=4..8`).
- **Batched W6/W4/WMMA concurrency kernels:** Unlocked pre-tiled FP16 WMMA on mixed and final-layer experts in all multi-request modes, added zero-spill batched `M <= 64` FP16-scaled W6/W4 WMMA for `QKVR`, `lm_head`, and DSpark `base_head`/`MLP`/`Markov`, and enabled both fitted weight banks across all launcher modes.
- **ROCr idle CPU fix preserved + smoke-test math check fixed:** Retains the v1.2.2 ROCr idle polling fix (`0.50–0.67%` idle CPU at 1M context) and fixes the math assertion in `scripts/smoke-test.py`. [v1.3.0 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.3.0)

### Previous releases
- **v1.2.2:** Bundled AMD's ROCr idle polling fix (`libhsa-runtime64.so.1`) for the pinned ROCm 10.0 SDK, lowering idle CPU usage at 1M context to ~1.4% of one core. [v1.2.2 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.2.2)
- **v1.2.1:** Enabled HIP blocking waits before ROCm initialization to prevent idle CPU pinning. [v1.2.1 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.2.1)
- **v1.2.0:** Defaulted text and vision launchers to a configurable **1,048,576-token** context window on 128 GB Strix Halo. [v1.2.0 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.2.0)
- **v1.1.0:** Added the native ROCm `gfx1151` vision encoder (`model.visual.*`) and `--mode vision` preset.
- **v1.0.0:** Initial Inkling 2 Mini CIRU Halo Agent release with RDNA 3.5 FP16 WMMA kernels, fitted expert weights, DSpark speculative decoding, and single-stream HIP graphs.

---

## Performance on AMD Ryzen AI Strix Halo (`gfx1151`)

Measured on an AMD Ryzen AI Max+ 395 with Radeon 8060S graphics (`20 WGPs / 40 CUs`) and 128 GB unified LPDDR5X memory (NixOS, Python 3.12.12, PyTorch `2.13.0+rocm10.0.0`, Triton `3.8.0`, vLLM `0.34.0+rocm100`).

### Real-world speed expectations (read before comparing)
Decode throughput on unified memory depends directly on **active context length**, **concurrent requests**, and **speculative draft acceptance**:
- **Short context (`≤ 4K`), single user, `T = 0` (coding/JSON/tools):** **~95–112 tok/s**.
- **Long agent context (`32K–72K` active KV history), single user:** **~45–63+ tok/s** (global KV-cache reads add ~9 ms/step at 64K; high-acceptance coding/tool turns at 72K measured **55.9–63.1 tok/s** even before v1.3.1's `K=7` vision fix).
- **Multi-agent bursts (`5–8` simultaneous requests):** Aggregate server output scales to **238–340 tok/s**, which divides across active streams (**~25–51 tok/s per request** depending on batch size, temperature, and prompt length). Agent harnesses that dispatch background summarization/routing requests alongside your main turn will split per-stream tok/s during those bursts.
- **High-temperature / high-entropy open-ended thinking (`T = 0.7–1.0`):** Lower draft acceptance (`~22–46%` vs. `65–93%` on code/tools) reduces tokens accepted per step; expect **~35–60 tok/s** during high-entropy reasoning at `B=1` and **~65–87 tok/s** on structured/coding prompts at `T = 0.7`.

### 1. Short-context single-stream and concurrent scaling (`≤ 4K` context, v1.3.x vs. v1.2.2)

| Serving profile (`--seqs 8` unless noted) | `B=1` | `B=2` (agg / req) | `B=4` (agg / req) | `B=8` (agg / req) | Gain vs. v1.2.2 (`B=8`) |
|---|---:|---:|---:|---:|---:|
| v1.0–v1.2.2 `concurrent` (eager, `K=7`) | 74.13 *(single)* | — | — | 88.92 / 11.12 | 1.00× |
| **v1.3.x Adaptive (`K=7 @ B=1`, `K=4 @ B=2..8`, `T=0`, default in v1.3.1)** | **100.23 tok/s** | 93.42 / 46.71 | **203.42 / 50.85** | **310.20 / 38.78** | **3.49×** |
| **v1.3.x Fixed `K=4` (`T=0`)** | 77.04 tok/s | 89.01 / 44.50 | 201.83 / 50.46 | **316.64 / 39.58** | **3.56×** |
| **v1.3.x `--mode sampling` (probabilistic `K=4`, `T=0.7`)** | **86.52 tok/s** | **101.36 / 50.68** | 136.01 / 34.00 | **238.73 / 29.84** | **2.68×** |
| v1.3.x `--mode single` (`K=7`, `T=0`) | **100.68 tok/s** *(peak `112.3` JSON)* | — | — | — | — |

On structured JSON (`order` fulfillment, `--context 4096`, `T=0`), adaptive `K=7/K=4` reaches **112.31 tok/s** at `B=1` and **340.10 tok/s wall (`409.05` agg decode, `51.13/req`, 8/8 exact JSON)** at `B=8`.

### 2. Smooth 512 → 64K prompt processing and long-context decode (`B=1`, CIRU-optimized weights)

| Prompt tokens | v1.0–v1.2.2 PP | **v1.3.x PP** | **v1.3.x TTFT** | PP Gain | **v1.3.x Decode TG** | `--no-spec` (`K=0`) TG |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,279.6 tok/s | **1,279.6 tok/s** | 0.40 s | — | 74–100+ tok/s | — |
| 2,048 | 1,645.0 tok/s | **1,645.0 tok/s** | 1.24 s | — | 74–100+ tok/s | — |
| 3,840 | 1,691.3 tok/s | **1,691.3 tok/s** | 2.27 s | — | 74–100+ tok/s | 48.5 tok/s |
| 8,192 | 1,640.8 tok/s | **1,640.8 tok/s** | 4.99 s | — | 59.1–75.1 tok/s | — |
| 32,017 *(32K)* | 1,248.3 tok/s | **1,416.9 tok/s** | **22.60 s** | **+13.5%** | **47.6–63.7 tok/s** | 36.9 tok/s |
| 47,977 *(48K)* | — | **1,324.2 tok/s** | 36.23 s | — | **59.4 tok/s** | — |
| **63,987 *(64K)*** | 788.5 tok/s | **1,300.3 tok/s** | **49.21 s** | **+64.9%** | **45.0–54.1 tok/s** | 30.7 tok/s |

Speculative decoding stays a net win at 62K–64K (`+44–45%` TG at `T=0`, `+34%` at `T=0.7` vs. `--no-spec` `30.7 tok/s`) because each verification step amortizes the 2.15 GB global KV-cache read across ~3 accepted tokens.

### 3. Vision and real multi-agent workload (8-agent AMD Logo 3D Studio)

Tested end-to-end on an autonomous **8-agent vision + tool-calling 3D scene studio** (`T=0.7`, prefix caching, native ROCm vision encoder, strict tool calling, probabilistic `K=4` baseline in v1.3.0):

| Metric | v1.1 / v1.2.2 (`vision` eager) | **v1.3.0 (`vision` FULL HIP graphs, `K=4`)** | Improvement |
|---|---:|---:|---:|
| **Pooled per-request TG** | 17.38 tok/s | **25.16 tok/s** *(peak `53.1 tok/s/req`)* | **+44.8%** |
| **Whole-workflow output rate** *(incl. vision, PP, tools, renders)* | 77.83 tok/s | **92.5 tok/s** | **+18.8%** |
| **Prefix cache hit rate / Draft acceptance** | — | **73.3% cache** / **62.9% accept (`3.51` tok/step)** | 100% FULL HIP graph |
| **Single-image PNG TTFT / TG (1M default context)** | — | **0.32 s** / **67.69 tok/s** | **0.50% idle CPU** |

*(In v1.3.1, `--mode vision` upgrades further to the full adaptive `K=7 @ B=1 / K=4 @ B=2..8` + fused W4 Markov + W8 draft + fused M8 loop stack, eliminating the v1.3.0 `B=1` vision slowdown.)*

### 4. Original Thinking Machines Lab checkpoint (`--original-weights`)

The same runtime serves the unmodified Thinking Machines Lab checkpoint mirrored in this repository without fitted weight banks. With `--original-weights`, a 3,900-token prompt reached **1,494.6 input tok/s** (`2.61 s` TTFT), and single-request decode reached **84.12 tok/s** on structured JSON, **79.73 tok/s** on native tool calling, and **69.79 tok/s** on Python coding.

---

## Install, upgrade, and run

Install or upgrade the ROCm vLLM runtime from the companion GitHub repository. The installation guide covers GPU access, pinned ROCm 10 wheels, the checked-in installer, upgrade steps from v1.2.x/v1.3.0, launch commands for every mode (`vision`, `concurrent`, `single`, `long-context`, `sampling`), and smoke-test verification. [GitHub runtime](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151) · [Installation guide](INSTALL.md) · [v1.3.1 release](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151/releases/tag/v1.3.1)

This Hugging Face repository contains the unmodified upstream model and DSpark draft checkpoint, the separate CIRU fitted weight banks under `optimized-weights/`, and the full v1.3.1 runtime bundle. Download the release with the Hugging Face CLI:

```bash
hf download jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151 --local-dir ./weights
```

**Upgrading from v1.3.0 (or v1.2.x):** Model and fitted weight files (`WEIGHTS.sha256`) are unchanged. Fetch tag `v1.3.1` in the GitHub runtime repository, install into a new prefix (`~/.local/opt/inkling-halo-v1.3.1`), and restart the server pointing at your existing `--model` directory.

Quick launch examples after installing:

```bash
# Default all-in-one server (vision + prefix cache + 8 slots + 1M ctx + adaptive K=7@B=1 / K=4@B=2..8)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --tools

# Text-only adaptive multi-agent server (same fast K=7/K=4 stack without vision encoder)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode concurrent --tools

# Dedicated single-stream server (1 slot, K=7, FULL HIP graph)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode single --tools

# 64K long-context preset (28 GiB KV cache, hybrid global prefill)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode long-context --context 66048 --tools

# Explicit probabilistic K=4 draft proposals (optional for high-temperature stochastic drafting)
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode sampling --seqs 8 --vision --prefix-cache --tools
```

Use `--context` to set a different request limit and `--kv-mib` to set the KV cache budget. Context covers prompt, image, and generated tokens together, and active requests share one KV cache.

---

## Credits and license

- Thinking Machines Lab developed Inkling 2 Mini and supplied the original checkpoint and DSpark draft weights. [Upstream model](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)
- CIRU developed the Strix Halo runtime integration, tuned RDNA 3.5 kernels, adaptive speculative HIP graphs, and optimized expert weights. The runtime builds on vLLM's Inkling support, Hugging Face Transformers, Triton, PyTorch, and AMD ROCm. [vLLM](https://github.com/vllm-project/vllm) · [Transformers](https://github.com/huggingface/transformers) · [Triton](https://github.com/triton-lang/triton) · [PyTorch](https://pytorch.org/) · [AMD ROCm](https://github.com/ROCm)
- The model weights and CIRU runtime are provided under Apache 2.0. [License](LICENSE) · [Notice](NOTICE)
