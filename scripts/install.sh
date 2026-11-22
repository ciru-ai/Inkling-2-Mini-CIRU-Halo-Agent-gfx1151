#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vllm_repo=https://github.com/ciru-ai/vllm.git
vllm_revision=7c4aa8b5b1d9aed94f4bfadcc38593a1f782066a
transformers_repo=https://github.com/huggingface/transformers.git
transformers_revision=6173a2f679ff4ed393afc499aefc569da7e68bc1
model_repo=jcbtc/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151

usage() {
    cat <<'HELP'
Usage: scripts/install.sh --venv PATH --transformers-deps PATH --prefix PATH [--download-model PATH]
       scripts/install.sh --verify-only [--venv PATH [--transformers-deps PATH]]
       scripts/install.sh --vllm-source PATH

Install the Inkling 2 Mini CIRU Halo Agent vLLM source in a separate prefix.
The existing ROCm virtual environment supplies its dependencies and native
extensions. An optional Hugging Face download verifies all model file hashes.

  --venv PATH            Python 3.12 venv with the validated ROCm vLLM wheel
                          and Python dependencies. See INSTALL.md.
  --transformers-deps PATH
                         Directory with the pinned Transformers dependencies
  --prefix PATH          New directory for the runtime and its launcher
  --download-model PATH  Download and verify model and fitted weights here
  --verify-only          Check bundle integrity and, if given, wheel versions
  --vllm-source PATH     Apply the runtime patch to a checkout at the base revision
  -h, --help            Show help
HELP
}

die() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 2
}

venv=
transformers_deps=
prefix=
source_tree=
model_dir=
verify_only=0
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --verify-only) verify_only=1; shift ;;
        --venv|--transformers-deps|--prefix|--vllm-source|--download-model)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || die "missing value for $1"
            case "$1" in
                --venv) venv="$2" ;;
                --transformers-deps) transformers_deps="$2" ;;
                --prefix) prefix="$2" ;;
                --vllm-source) source_tree="$2" ;;
                --download-model) model_dir="$2" ;;
            esac
            shift 2
            ;;
        *) die "unknown option: $1" ;;
    esac
done

if (( verify_only )); then
    [[ -z "$prefix$source_tree$model_dir" ]] || die '--verify-only accepts only --venv and --transformers-deps'
    [[ -z "$transformers_deps" || -n "$venv" ]] || die '--transformers-deps requires --venv'
elif [[ -n "$source_tree" ]]; then
    [[ -z "$prefix$venv$model_dir$transformers_deps" ]] || die '--vllm-source is a standalone option'
else
    [[ -n "$prefix" && -n "$venv" && -n "$transformers_deps" ]] ||
        die 'specify --prefix, --venv and --transformers-deps'
    [[ ! -e "$prefix" && ! -L "$prefix" ]] || die "prefix already exists: $prefix"
fi

(cd "$repo_root" && sha256sum --quiet -c MANIFEST.sha256)
printf '%s\n' 'Runtime bundle checksums match.'

if [[ -n "$venv" ]]; then
    [[ -x "$venv/bin/python" ]] || die "Python executable not found under $venv/bin"
    python_bin="$(cd -- "$venv/bin" && pwd)/python"
    site_vllm=$(PYTHONPATH= "$python_bin" - <<'PY'
import importlib.metadata
import sys
from importlib.metadata import version
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Use CPython 3.12 for the vLLM 0.34.0 ROCm wheel.")
expected = {
    "vllm": "0.34.0+rocm100",
    "torch": "2.13.0+rocm10.0.0",
    "triton": "3.8.0+git669b31ac",
    "huggingface-hub": "2.4.0",
    "safetensors": "0.8.0",
}
for package, required in expected.items():
    try:
        installed = version(package)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f"{package} is missing. Follow INSTALL.md.") from None
    if installed != required:
        raise SystemExit(f"{package} {installed} is installed. Expected {required}.")
site = Path(importlib.metadata.distribution("vllm").locate_file("vllm")).resolve()
if not (site / "_rocm_C.abi3.so").is_file():
    raise SystemExit(f"ROCm native extension is missing from {site}.")
print(site)
PY
)
    printf '%s\n' 'Python and ROCm wheel versions match.'
fi

if [[ -n "$transformers_deps" ]]; then
    [[ -d "$transformers_deps" ]] || die "Transformers dependencies not found at $transformers_deps"
    transformers_deps="$(cd -- "$transformers_deps" && pwd)"
    PYTHONPATH= "$python_bin" - "$transformers_deps" <<'PY'
import importlib.metadata
import sys

expected = {"huggingface-hub": "1.31.0", "safetensors": "0.8.0"}
installed = {
    dist.metadata["Name"].lower().replace("_", "-"): dist.version
    for dist in importlib.metadata.distributions(path=[sys.argv[1]])
}
for package, required in expected.items():
    if installed.get(package) != required:
        raise SystemExit(f"Install {package}=={required} into {sys.argv[1]} using INSTALL.md.")
print("Pinned Transformers dependencies match.")
PY
fi

if (( verify_only )); then
    exit 0
fi

if [[ -n "$source_tree" ]]; then
    base_revision=b426b46e63408c41b0aec069fdaf24dd640abe58
    [[ "$(git -C "$source_tree" rev-parse HEAD)" == "$base_revision" ]] ||
        die "the source checkout must be at $base_revision"
    git -C "$source_tree" apply --check "$repo_root/patches/vllm-inkling2-gfx1151.patch"
    git -C "$source_tree" apply "$repo_root/patches/vllm-inkling2-gfx1151.patch"
    exit 0
fi

if [[ -n "$model_dir" ]]; then
    PYTHONPATH= "$python_bin" - "$model_repo" "$model_dir" "$repo_root/WEIGHTS.sha256" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id, destination, manifest = sys.argv[1:]
files = [line.split(maxsplit=1)[1] for line in Path(manifest).read_text().splitlines()]
snapshot_download(repo_id=repo_id, local_dir=destination, allow_patterns=files)
PY
    (cd "$model_dir" && sha256sum --quiet -c "$repo_root/WEIGHTS.sha256")
    printf '%s\n' 'Base checkpoint and fitted weight checksums match.'
fi

mkdir -p -- "$(dirname -- "$prefix")"
prefix="$(cd -- "$(dirname -- "$prefix")" && pwd)/$(basename -- "$prefix")"
stage=$(mktemp -d "${prefix}.install.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT

git init --quiet "$stage/vllm-src"
git -C "$stage/vllm-src" fetch --quiet --depth 1 "$vllm_repo" "$vllm_revision"
git -C "$stage/vllm-src" checkout --quiet --detach FETCH_HEAD
[[ "$(git -C "$stage/vllm-src" rev-parse HEAD)" == "$vllm_revision" ]] ||
    die 'fetched vLLM revision mismatch'
git init --quiet "$stage/transformers-src"
git -C "$stage/transformers-src" fetch --quiet --depth 1 "$transformers_repo" "$transformers_revision"
git -C "$stage/transformers-src" checkout --quiet --detach FETCH_HEAD
[[ "$(git -C "$stage/transformers-src" rev-parse HEAD)" == "$transformers_revision" ]] ||
    die 'fetched Transformers revision mismatch'
cp -a "$repo_root/overlay/vllm/." "$stage/vllm-src/vllm/"
for support_file in "$site_vllm"/*.so "$site_vllm/_version.py" "$site_vllm/vllm-rs"; do
    if [[ -e "$support_file" ]]; then
        ln -s -- "$support_file" "$stage/vllm-src/vllm/"
    fi
done

cp -a "$repo_root/scripts" "$stage/scripts"
cp -a "$repo_root/LICENSE" "$repo_root/NOTICE" "$stage/"
{
    printf '#!/usr/bin/env bash\n'
    printf 'exec %q --overlay %q --transformers-src %q --transformers-deps %q --python %q "$@"\n' \
        "$prefix/scripts/serve.sh" "$prefix/vllm-src" "$prefix/transformers-src/src" "$transformers_deps" "$python_bin"
} > "$stage/serve.sh"
chmod +x "$stage/serve.sh"
mv -T --no-clobber -- "$stage" "$prefix"
[[ ! -e "$stage" ]] || die "prefix was created during installation: $prefix"
printf 'Installed. Start with %q --model /path/to/model --tools\n' "$prefix/serve.sh"
