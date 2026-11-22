# Source this from the launch script before importing Torch.

ciru_use_fixed_rocr() {
    local python=$1 sdk original override_dir fixed version preload=${LD_PRELOAD:-}
    local affected_sha=815364af872593e577206ff22a29141af63e800a78488b2ee02cf04532e14a46
    local repaired_sha=61732d2d4ea4c4d0ee51738175d54dca5cb24c9e3e0a6a4a9da5998ee96d3845
    sdk=$("$python" -I -c '
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("_rocm_sdk_core")
if spec and spec.origin:
    print(Path(spec.origin).parent / "lib")
') || return
    [[ -n "$sdk" ]] || return 0
    original="$sdk/libhsa-runtime64.so.1"
    [[ -f "$original" ]] || return 0
    version=$(sha256sum -- "$original") || return
    [[ "${version%% *}" == "$affected_sha" ]] || return 0

    override_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return
    fixed="$override_dir/libhsa-runtime64.so.1"
    if [[ "$fixed" =~ [[:space:]:] ]]; then
        printf 'Ciru ROCm: invalid path for LD_PRELOAD: %s\n' "$fixed" >&2
        return 1
    fi
    version=$(sha256sum -- "$fixed") || return
    if [[ "${version%% *}" != "$repaired_sha" ]]; then
        printf 'Ciru ROCm: repaired library does not match its checksum\n' >&2
        return 1
    fi
    case " ${preload//:/ } " in
        *" $fixed "*) ;;
        *) export LD_PRELOAD="$fixed${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
    export LD_LIBRARY_PATH="$sdk:$sdk/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$python" -I - "$fixed" <<'PY'
from pathlib import Path
import sys

fixed = Path(sys.argv[1]).resolve()
mapped = {
    Path(fields[5])
    for line in Path("/proc/self/maps").read_text().splitlines()
    if len(fields := line.split(maxsplit=5)) == 6
    and "/libhsa-runtime64.so" in fields[5]
}
if mapped != {fixed}:
    raise SystemExit("Ciru ROCm: the corrected library could not be loaded. Use a compatible Linux runtime such as Ubuntu 24.04.")
PY
}
