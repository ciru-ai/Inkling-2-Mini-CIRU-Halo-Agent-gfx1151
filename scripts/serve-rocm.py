# SPDX-License-Identifier: Apache-2.0
"""Set HIP blocking waits before vLLM or a spawned worker initializes ROCm."""

import ctypes
from importlib.util import find_spec
from pathlib import Path


def set_hip_blocking_waits():
    sdk = find_spec("_rocm_sdk_core")
    if sdk is None or sdk.origin is None:
        raise SystemExit("ROCm SDK wheel missing. Install the pinned wheels in INSTALL.md.")

    hip_path = Path(sdk.origin).parent / "lib" / "libamdhip64.so.7"
    try:
        hip = ctypes.CDLL(str(hip_path))
    except OSError as exc:
        raise SystemExit(f"Cannot load the vLLM environment's HIP runtime: {exc}") from exc

    set_device_flags = hip.hipSetDeviceFlags
    set_device_flags.argtypes = [ctypes.c_uint]
    set_device_flags.restype = ctypes.c_int
    status = set_device_flags(0x4)  # hipDeviceScheduleBlockingSync
    if status != 0:
        raise SystemExit(f"Cannot set HIP blocking waits before vLLM startup (HIP status {status}).")


# multiprocessing spawn executes the launcher as __mp_main__ in each worker.
# Apply the setting before the worker unpickles and imports vLLM.
if __name__ in ("__main__", "__mp_main__"):
    set_hip_blocking_waits()

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
