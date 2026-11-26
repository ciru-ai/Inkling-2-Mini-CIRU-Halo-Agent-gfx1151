# Install Inkling 2 Mini CIRU Halo Agent (gfx1151)

The release accepts text and image inputs and serves text, reasoning and native tool calls through an OpenAI-compatible API. It uses ROCm on the Radeon 8060S in the AMD Ryzen AI Max+ 395. The performance measurements used the 128 GB unified-memory configuration on NixOS with CPython 3.12.12, vLLM 0.34.0 ROCm 10, PyTorch 2.13.0 ROCm 10 and Triton 3.8.0. Ubuntu 24.04 LTS is a setup path for the same Linux wheel. The benchmark runs used NixOS.

Budget at least 56 GB of free disk space for the model, optional DFlash 2 drafter (~1 GB), ROCm Python environment and Triton cache. The model download includes about 23 GB of base and fitted weights.

## 1. Prepare Linux and ROCm

Start with a working AMDGPU driver. On Ubuntu 24.04 LTS, install the client-side prerequisites and grant GPU device access if needed.

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential python3.12 python3.12-venv libnuma1
sudo usermod -aG render,video "$USER"
```

Sign out and sign back in after changing groups. Confirm that your user can access the GPU devices.

```bash
ls -l /dev/kfd /dev/dri/renderD*
test -r /dev/kfd && test -w /dev/kfd
```

The public vLLM wheel uses TheRock's ROCm Python packages. Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to set up an isolated environment and resolve the ROCm wheels from the [pinned vLLM index](https://wheels.vllm.ai/rocm/0.34.0/rocm100/).

```bash
uv venv --python 3.12 "$HOME/.venvs/inkling-halo"
INKLING_PYTHON="$HOME/.venvs/inkling-halo/bin/python"
INKLING_DEPS="$HOME/.local/share/inkling-halo/transformers-deps"
uv pip install --python "$INKLING_PYTHON" \
  --extra-index-url https://wheels.vllm.ai/rocm/0.34.0/rocm100/ \
  'vllm==0.34.0+rocm100' 'huggingface-hub==2.4.0' 'safetensors==0.8.0'
uv pip install --python "$INKLING_PYTHON" --target "$INKLING_DEPS" --no-deps \
  'huggingface-hub==1.31.0' 'safetensors==0.8.0'
"$INKLING_PYTHON" -c 'import torch; p = torch.cuda.get_device_properties(0); print(torch.version.hip, p.name, p.gcnArchName)'
```

The last command should display a HIP version and an architecture beginning with `gfx1151`. The standalone launcher gives the pinned Transformers source its own dependency directory. The wheel and its package metadata stay intact in the virtual environment.

## 2. Install the runtime and retrieve weights

```bash
git clone https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151.git
cd Inkling-2-Mini-CIRU-Halo-Agent-gfx1151
INKLING_ROOT="$HOME/.local/opt/inkling-halo-v1.6.0"
INKLING_MODEL="$HOME/models/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151"

bash scripts/install.sh --venv "$HOME/.venvs/inkling-halo" \
  --transformers-deps "$HOME/.local/share/inkling-halo/transformers-deps" \
  --prefix "$INKLING_ROOT" --download-model "$INKLING_MODEL"
```

The installer verifies the package and model checksums, retrieves the pinned [vLLM runtime](https://github.com/ciru-ai/vllm/commit/b1dcc9e6f4b1fadc3a6a21c96b8e449de81238d6) and [Transformers support](https://github.com/huggingface/transformers/commit/6173a2f679ff4ed393afc499aefc569da7e68bc1), installs the bundled portable `gfx1151` ABI v3 all-HIP prefill & decode library (`scripts/rocm/hip_inkling_gfx1151_07458d24acf9.so`, requiring no `hipcc` or `rocm-sdk-devel`), and links the ROCm extensions into the standalone prefix. The launcher reuses the virtual environment. If you already have the complete [Hugging Face bundle](https://huggingface.co/jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151), omit `--download-model` and point the launcher at your copy.

You can also run `bash scripts/install.sh` from the root of the complete Hugging Face download.

### Upgrading from v1.2.x–v1.5.0 to v1.6.0

Your existing model and fitted weight files (`WEIGHTS.sha256`) are unchanged and do not need to be re-downloaded. Update the runtime repo, install into the new prefix, and restart the server:

```bash
cd Inkling-2-Mini-CIRU-Halo-Agent-gfx1151
git fetch origin --tags
git switch --detach v1.6.0
INKLING_ROOT="$HOME/.local/opt/inkling-halo-v1.6.0"

bash scripts/install.sh --venv "$HOME/.venvs/inkling-halo" \
  --transformers-deps "$HOME/.local/share/inkling-halo/transformers-deps" \
  --prefix "$INKLING_ROOT"
```

Use `bash scripts/install.sh --verify-only` to check package integrity without downloading or installing anything. Add `--venv "$HOME/.venvs/inkling-halo"` to check the wheel versions and ROCm extension. Add `--transformers-deps "$HOME/.local/share/inkling-halo/transformers-deps"` with that virtual environment to verify the isolated dependencies.

## 3. Serve

### Recommended (v1.6.0 fastest): Inco AI DFlash 2 drafter (`--dflash2`)

Download the open-weights DFlash 2 block-diffusion drafter (`~1 GB`) and pass `--dflash2` to enable native RDNA 3.5 W6 WMMA DFlash 2 (`K=7` block-8 drafting, `TRITON_ATTN` 3D split-KV sliding-window attention, lossless probabilistic drafting for both `T=0` and `T>0`, FULL Target + Draft HIP graphs, native ROCm vision, prefix caching, native tool calling, 8 concurrent sequences, and 1,048,576-token context in 48 GiB BF16 KV):

```bash
INKLING_DFLASH2="$HOME/models/Inkling-2-Mini-DFlash2"
hf download incoai/Inkling-2-Mini-DFlash2 --local-dir "$INKLING_DFLASH2"

"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --dflash2 "$INKLING_DFLASH2" --kv-mib 49152 --tools
```

With `--dflash2`, `serve.sh` uses fixed `K=7` across `B=1..8` and sets `draft_sample_method="probabilistic"` (requests with `temperature: 0` still run the exact greedy path, while `temperature > 0` requests use exact probability-ratio rejection sampling; set `INKLING_DFLASH2_SAMPLE_METHOD=greedy` to force one-hot greedy drafting).

### Self-contained default (no extra drafter download): bundled adaptive DSpark

Without `--dflash2`, `serve.sh` starts the self-contained all-in-one server with native RDNA 3.5 all-HIP prefill and decode acceleration, native ROCm vision, prefix caching, native tool calling, 8 concurrent sequences, 1,048,576-token context (52 GiB BF16 KV cache), FULL Target + Draft HIP graphs, and dual-depth adaptive DSpark speculation (`K=7` at `B=1`, `K=4` at `B=2..8`). Fitted weights are enabled automatically when the model directory includes them, and greedy draft proposals support both `temperature = 0` and `temperature > 0` losslessly via exact rejection sampling.

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --tools
```

In another terminal, check the server and run the reasoning and tool-calling smoke test.

```bash
curl -fsS http://127.0.0.1:18753/health
curl -fsS http://127.0.0.1:18753/v1/models
python3 scripts/smoke-test.py --check-tools
```

The API base URL is `http://127.0.0.1:18753/v1` and the model ID is `main`. The first launch builds a local Triton kernel cache and may take longer. Use `--dry-run` to preview the selected launch settings.

Start the server through `serve.sh` to load both the prebuilt native `gfx1151` ABI v3 all-HIP prefill & decode library (`hip_inkling_gfx1151_07458d24acf9.so`) and the corrected ROCr library for the pinned ROCm 10.0 SDK. The launcher checks the SDK and library checksums. On multi-GPU hosts, use `ROCR_VISIBLE_DEVICES` to expose only the Strix Halo GPU. [AMD ROCr fix](https://github.com/ROCm/rocm-systems/pull/7898)

#### Harmless startup notices

You may see the following upstream vLLM notices during startup—all are benign on Strix Halo (`gfx1151`):

- `Module torchcodec.decoders was found but failed to import` (`No module named 'torchcodec'`): upstream vLLM probes optional `torchcodec` for audio/video decoding. Inkling 2 Mini uses PIL/torchvision for images, so `torchcodec` is not needed.
- `tvm_ffi/_optional_torch_c_dlpack.py ... UserWarning: Failed to JIT torch c dlpack extension`: `apache-tvm-ffi` is a transitive dependency used only by NVIDIA CuTeDSL kernels and is unused by Inkling's AMD ROCm HIP/Triton stack. (Installing `torch-c-dlpack-ext` in the venv silences the warning if desired.)
- `Default vLLM sampling parameters have been overridden by the model's generation_config.json: {'temperature': 1.0}`: Thinking Machines Lab's official Inkling 2 Mini checkpoint defaults to `temperature: 1.0` when clients omit `temperature`. Requests passing `"temperature": 0` still run deterministic greedy decoding.

Set `--context` to override the request limit. The launcher sizes the KV pool from that limit unless you set `--kv-mib` explicitly. Choose a smaller context and cache budget on systems with less available memory.

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --context 32768 --kv-mib 24576 --tools
```

For maximum throughput on long `8K–64K` prompts (trading ~135 tok/s on short `2K–4K` prompts for `+40–65 tok/s` across `8K–58K`), pass `--batch-tokens 8192`:

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --batch-tokens 8192 --tools
```

### Send an image to the default server

Because native ROCm vision is enabled in the default `vision` preset with zero text-turn penalty, you can send images directly to the default server above. For a first check, use a small local PNG or JPEG such as 512 × 512:

```bash
python3 - ./example.png <<'PYTHON'
import base64
import json
import mimetypes
from pathlib import Path
import sys
from urllib.request import Request, urlopen

image = Path(sys.argv[1])
mime = mimetypes.guess_type(image.name)[0]
if mime not in ("image/png", "image/jpeg"):
    raise SystemExit("Choose a PNG or JPEG image.")
encoded = base64.b64encode(image.read_bytes()).decode("ascii")
payload = {
    "model": "main",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            {"type": "text", "text": "Describe the main subject and its color in one sentence."},
        ],
    }],
    "temperature": 0,
    "max_tokens": 256,
}
request = Request(
    "http://127.0.0.1:18753/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=120) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
PYTHON
```

### Optional presets

- **Text-only adaptive multi-agent (`--mode concurrent`):** Identical fast adaptive `K=7 @ B=1 / K=4 @ B=2..8` 8-sequence stack without loading the vision encoder or enabling prefix caching by default:
  ```bash
  "$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode concurrent --tools
  ```
- **Dedicated single-stream (`--mode single`):** 1 sequence (`--seqs 1`), `K=7`, text-only:
  ```bash
  "$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode single --tools
  ```
- **64K long-context preset (`--mode long-context`):** 1 sequence with a 28 GiB BF16 KV pool and hybrid AOTriton-Far + Triton-Near global prefill:
  ```bash
  "$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode long-context --context 66048 --tools
  ```
- **Explicit probabilistic `K=4` drafting (`--mode sampling`):** Opts into full-vocab stochastic draft proposals (`draft_sample_method="probabilistic"`, fixed `K=4`, disables the `B=1` fused greedy Markov fast path). Add `--vision --prefix-cache` if combining probabilistic draft proposals with images:
  ```bash
  "$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode sampling --seqs 8 --context 32768 --prefix-cache --tools
  ```

Pass `--original-weights` to use only the original Thinking Machines Lab checkpoint files without the fitted weight banks. Pass `--no-graphs` to fall back to eager execution if needed for debugging.
