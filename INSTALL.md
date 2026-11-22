# Install Inkling 2 Mini CIRU Halo Agent (gfx1151)

The release serves text, reasoning and native tool calls through an OpenAI-compatible API. It uses ROCm on the Radeon 8060S in the AMD Ryzen AI Max+ 395. The performance measurements used the 128 GB unified-memory configuration on NixOS with CPython 3.12.12, vLLM 0.34.0 ROCm 10, PyTorch 2.13.0 ROCm 10 and Triton 3.8.0. Ubuntu 24.04 LTS is a setup path for the same Linux wheel. The benchmark runs used NixOS.

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
INKLING_ROOT="$HOME/.local/opt/inkling-halo"
INKLING_MODEL="$HOME/models/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151"

bash scripts/install.sh --venv "$HOME/.venvs/inkling-halo" \
  --transformers-deps "$HOME/.local/share/inkling-halo/transformers-deps" \
  --prefix "$INKLING_ROOT" --download-model "$INKLING_MODEL"
```

The installer verifies the package and model checksums, retrieves the pinned [vLLM runtime](https://github.com/ciru-ai/vllm/commit/7c4aa8b5b1d9aed94f4bfadcc38593a1f782066a) and [Transformers support](https://github.com/huggingface/transformers/commit/6173a2f679ff4ed393afc499aefc569da7e68bc1), and links the ROCm extensions into the standalone prefix. The launcher reuses the virtual environment. If you already have the complete [Hugging Face bundle](https://huggingface.co/jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151), omit `--download-model` and point the launcher at your copy.

Use `bash scripts/install.sh --verify-only` to check package integrity without downloading or installing anything. Add `--venv "$HOME/.venvs/inkling-halo"` to check the wheel versions and ROCm extension. Add `--transformers-deps "$HOME/.local/share/inkling-halo/transformers-deps"` with that virtual environment to verify the isolated dependencies.

## 3. Serve

Start one 4,096-token stream with DSpark and native tool calling. Fitted weights are enabled automatically when the model directory includes them.

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

### Serve a 64K prompt

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode long-context --tools
```

This preset allows 66,048 total tokens for the prompt and completion, with a 28 GiB BF16 KV pool. It uses both sets of fitted weights.

### Serve eight requests

```bash
"$INKLING_ROOT/serve.sh" --model "$INKLING_MODEL" --mode concurrent --tools
```

This preset uses eight sequences at a 4,096-token request limit with a shared 6 GiB BF16 KV pool. It uses the fitted routed experts, matching the measured concurrent run.

Pass `--original-weights` to use the original checkpoint files with the AMD runtime. The published highlights using fitted weights assume the default behavior.
