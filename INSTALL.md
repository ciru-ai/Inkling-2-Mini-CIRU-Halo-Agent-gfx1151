# Install Inkling 2 Mini CIRU Halo Agent (gfx1151)

The release accepts text and image inputs and serves text, reasoning and native tool calls through an OpenAI-compatible API. It uses ROCm on the Radeon 8060S in the AMD Ryzen AI Max+ 395. The performance measurements used the 128 GB unified-memory configuration on NixOS with CPython 3.12.12, vLLM 0.34.0 ROCm 10, PyTorch 2.13.0 ROCm 10 and Triton 3.8.0. Ubuntu 24.04 LTS is a setup path for the same Linux wheel. The benchmark runs used NixOS.

Budget at least 55 GB of free disk space for the model, ROCm Python environment and Triton cache. The model download includes about 23 GB of base and fitted weights.

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
INKLING_ROOT="$HOME/.local/opt/inkling-halo-v1.3.0"
INKLING_MODEL="$HOME/models/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151"

bash scripts/install.sh --venv "$HOME/.venvs/inkling-halo" \
  --transformers-deps "$HOME/.local/share/inkling-halo/transformers-deps" \
  --prefix "$INKLING_ROOT" --download-model "$INKLING_MODEL"
```

The installer verifies the package and model checksums, retrieves the pinned [vLLM runtime](https://github.com/ciru-ai/vllm/commit/c95dbfa9cf4e6cbe94f9691e9febc868a217ecd7) and [Transformers support](https://github.com/huggingface/transformers/commit/6173a2f679ff4ed393afc499aefc569da7e68bc1), and links the ROCm extensions into the standalone prefix. The launcher reuses the virtual environment. If you already have the complete [Hugging Face bundle](https://huggingface.co/jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151), omit `--download-model` and point the launcher at your copy.

You can also run `bash scripts/install.sh` from the root of the complete Hugging Face download.

### Upgrading from v1.2.x to v1.3.0

Your existing model and fitted weight files (`WEIGHTS.sha256`) are unchanged and do not need to be re-downloaded. Update the runtime repo, install into the new prefix, and restart the server:

```bash
cd Inkling-2-Mini-CIRU-Halo-Agent-gfx1151
git fetch origin --tags
git switch --detach v1.3.0
INKLING_ROOT="$HOME/.local/opt/inkling-halo-v1.3.0"

bash scripts/install.sh --venv "$HOME/.venvs/inkling-halo" \
  --transformers-deps "$HOME/.local/share/inkling-halo/transformers-deps" \
  --prefix "$INKLING_ROOT"
```

Use `bash scripts/install.sh --verify-only` to check package integrity without downloading or installing anything. Add `--venv "$HOME/.venvs/inkling-halo"` to check the wheel versions and ROCm extension. Add `--transformers-deps "$HOME/.local/share/inkling-halo/transformers-deps"` with that virtual environment to verify the isolated dependencies.

## 3. Serve

Start a single stream with DSpark and native tool calling. The launcher defaults to 1,048,576 tokens and reserves 52 GiB for the BF16 KV cache. Fitted weights are enabled automatically when the model directory includes them.

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

Start the server through `serve.sh` to load the corrected ROCr library for the pinned ROCm 10.0 SDK. The launcher checks the SDK and library checksums. The backport was built on Ubuntu 24.04 LTS and tested on NixOS with one visible Strix Halo GPU. On multi-GPU hosts, use `ROCR_VISIBLE_DEVICES` to expose only the Strix Halo GPU. [AMD ROCr fix](https://github.com/ROCm/rocm-systems/pull/7898)

Set `--context` to override the request limit. The launcher sizes the KV pool from that limit unless you set `--kv-mib` explicitly. Choose a smaller context and cache budget on systems with less available memory.

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --context 32768 --kv-mib 24576 --tools
```

### Serve concurrent agents (adaptive single-stream and 8-way throughput)

Version 1.3.0 upgrades `--mode concurrent` with multi-request FULL Target + Draft HIP graphs (`1..8` active sequences), batched W6/W4/WMMA projections, and dual-depth adaptive DSpark speculation (`K=7` for a single active request, switching automatically to `K=4` for 2–8 concurrent requests):

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode concurrent --tools
```

Pass `--context 32768 --prefix-cache` for a compact 24 GiB KV pool, `--spec-tokens 4` to fix a four-token draft depth across all batch sizes, or `--vision` to enable image input alongside greedy concurrent serving.

### Serve images and multi-agent vision workflows

Start the `vision` preset. In v1.3.0, it runs with FULL Target + Draft HIP graphs (`1..8` sequences), four-token probabilistic DSpark speculation (`K=4`), prefix caching, and the default 1,048,576-token window (52 GiB BF16 KV cache):

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode vision --tools
```

For a first check, use a small local PNG or JPEG such as 512 × 512. The example sends the image and a text question in one request.

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

### Serve long prompts (32K–64K+ with hybrid global prefill)

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode long-context --context 66048 --tools
```

Version 1.3.0 routes global far-history prefill (`>= 2,048` far tokens) through native ROCm AOTriton FlashAttention-2 with an online-softmax near-window merge, eliminating the 64K prefill cliff (`+61.6%` faster at 62K–64K prompts) and improving 64K decode split-KV memory bandwidth. This example matches the 64K benchmark with 66,048 total tokens and a 28 GiB BF16 KV pool.

### Serve sampling (`T > 0`) workloads

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode sampling --seqs 8 --context 32768 --prefix-cache --tools
```

Uses probabilistic DSpark (`K=4`) with FULL Target + Draft HIP graphs for temperature-sampled generation across 1–8 sequences.

Pass `--original-weights` to use only the original Thinking Machines Lab checkpoint files without the fitted weight banks. Pass `--no-graphs` to fall back to eager execution if needed for debugging.
