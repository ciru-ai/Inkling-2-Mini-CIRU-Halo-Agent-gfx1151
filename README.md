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

Run Thinking Machines Lab's Inkling 2 Mini locally on **AMD Ryzen AI Max+ 395 with Radeon 8060S**. CIRU's ROCm vLLM build reached **1,691.3 prompt tokens/s**, **74.13 decode tokens/s across three single-request workloads**, and **88.92 aggregate output tokens/s across eight concurrent requests**. At a 65,536-token prompt, it reached **54.08 decode tokens/s**. The OpenAI-compatible API accepts text and images, streams reasoning, and supports native tool calling.

![Inkling 2 Mini CIRU Halo Agent benchmark on AMD Ryzen AI Strix Halo](assets/inklingamd.png)

*The artwork highlights the CIRU-optimized configuration on Strix Halo with seven-token DSpark speculative decoding.*

## Vision on Strix Halo

Version 1.1 adds a native ROCm vision encoder. Start the `vision` preset to send an image alongside a text question. The original checkpoint already contains the vision weights. [Vision setup and image request](INSTALL.md#serve-images)

## Text performance on AMD Ryzen AI Strix Halo

Measured on a Ryzen AI Max+ 395 with Radeon 8060S graphics and 128 GB of unified memory. The runtime uses RDNA 3.5 matrix cores to accelerate expert projections, tunes attention for long prompts, and uses HIP graphs for single-request decode.

### CIRU-optimized expert weights

| Prompt tokens | Prompt processing (tok/s) | Time to first token |
| ---: | ---: | ---: |
| 512 | 1,279.6 | 0.40 s |
| 2,048 | 1,645.0 | 1.24 s |
| 3,840 | **1,691.3** | 2.27 s |
| 8,192 | **1,640.8** | **4.99 s** |
| 32,768 | 1,248.3 | 26.25 s |
| 65,536 | 788.5 | 83.11 s |

| Generation workload | Completion throughput (tok/s) |
| --- | ---: |
| Single request, pooled across structured output, math, and Python tasks | **74.13** |
| Single request with a 32K-token prompt | **63.69** |
| Single request with a 64K-token prompt | **54.08** |
| Eight concurrent requests, aggregate over the full batch | **88.92** |

Prompt throughput divides input length by time to first token. Single-request decode is measured after the first token. The concurrent result measures total completion tokens over the time for all eight requests to finish. The single-request measurements use both sets of fitted weights. The eight-request run uses the fitted expert files and the original final-layer weights.

### Original Thinking Machines Lab checkpoint

The same runtime serves the unmodified checkpoint mirrored here. With that configuration, a 3,900-token prompt reached **1,494.6 input tok/s** and **2.61 s** to first token. Single-request decode reached **84.12 tok/s** on a structured JSON task, **79.73 tok/s** on native tool calling, and **69.79 tok/s** on a Python coding task.

## Install and run

Install the ROCm vLLM runtime from the companion GitHub repository. The installation guide covers GPU access, dependencies, the checked-in installer, launch commands for the original and optimized weights, and verification through the OpenAI-compatible API. [GitHub runtime](https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151) · [Installation guide](INSTALL.md)

The Hugging Face repository contains the original model and DSpark draft checkpoint plus the separate CIRU expert weights used for the benchmark artwork. Download the release with the Hugging Face CLI.

```bash
hf download jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151 --local-dir ./weights
```

The standard single-request preset uses a 4,096-token context. The long-context benchmark used a 66,048-token window to fit a 65,536-token prompt with room to generate. Context limits include input and output. The vision preset allows 32,768 total tokens, including the image tokens.

## Credits and license

- Thinking Machines Lab developed Inkling 2 Mini and supplied the original checkpoint and DSpark draft weights. [Upstream model](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)
- CIRU developed the Strix Halo runtime integration, tuned kernels, and optimized expert weights. The runtime builds on vLLM's Inkling support, Hugging Face Transformers, Triton, PyTorch, and AMD ROCm. [vLLM](https://github.com/vllm-project/vllm) · [Transformers](https://github.com/huggingface/transformers) · [Triton](https://github.com/triton-lang/triton) · [PyTorch](https://pytorch.org/) · [AMD ROCm](https://github.com/ROCm)
- The model weights and CIRU runtime are provided under Apache 2.0. [License](LICENSE) · [Notice](NOTICE)
