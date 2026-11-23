#!/usr/bin/env bash
# Portable launcher for Inkling 2 Mini CIRU Halo Agent (gfx1151) v1.4.0.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
    cat <<'EOF'
Usage: serve.sh --model PATH [OPTIONS]

Required:
  --model PATH            Local directory downloaded from the release on Hugging Face

Modes & Serving Options:
  --mode MODE             vision (default: image + text, prefix caching, 8 sequences, 1M context,
                                  FULL Target+Draft HIP graphs, adaptive DSpark K=7@B=1 / K=4@B=2..8,
                                  fused W4 Markov + W8 draft + fused M8 loop; lossless for any T>=0),
                          concurrent (same fast adaptive 8-sequence stack, text-only, no prefix cache by default),
                          single (single-stream FULL HIP graph, DSpark K=7, text-only),
                          long-context (single-stream FULL HIP graph, DSpark K=7, hybrid 64K+ global prefill),
                          sampling (explicit probabilistic DSpark K=4 draft proposals with FULL Target+Draft HIP graphs)
  --served-model-name ID  OpenAI API model name (default: main)
  --host HOST             Bind host (default: 127.0.0.1)
  --port PORT             Bind port (default: 18753)
  --context N             Max prompt + completion tokens (default: 1048576)
  --seqs N                Max concurrent sequences (default: 8 in vision/concurrent; 1 in single/long-context/sampling)
  --batch-tokens N        Scheduler max_num_batched_tokens (default: 4096)
  --prefill-chunk-tokens N
                          Optional --long-prefill-token-threshold cap
  --kv-mib N              BF16 KV cache budget in MiB (default: sized for context and concurrency)
  --spec-tokens K         Fixed DSpark draft depth 1..7 (overrides mode default)
  --adaptive-spec         Enable dual-depth adaptive DSpark HIP graphs (K=7 @ B=1, K=4 @ B=2..N; default in vision/concurrent)
  --no-adaptive-spec      Disable adaptive depth switching
  --no-spec               Disable DSpark speculative decoding
  --prefix-cache          Enable vLLM prefix caching (default: on in vision mode)
  --no-prefix-cache       Disable vLLM prefix caching
  --tools                 Enable Inkling native auto tool calling
  --vision                Enable native ROCm vision encoder in any mode (default in vision mode)
  --language-model-only   Disable vision encoder (default outside vision mode)
  --draft-graph           Enable DSpark draft HIP graph (default: on when graphs are enabled)
  --no-draft-graph        Disable DSpark draft HIP graph
  --no-graphs             Disable FULL HIP graph capture (run eager)

Runtime & Optional Quant Bank Paths:
  --python PATH           Python interpreter inside the vLLM 0.34 ROCm 10 environment
                          (auto-detected from INKLING_PYTHON, active VIRTUAL_ENV, or .venv)
  --overlay PATH          Path to the installed vLLM source tree
  --transformers-src PATH Path to the pinned Transformers src directory
  --transformers-deps PATH
                          Path to the isolated Transformers dependencies
  --triton-cache PATH     Triton kernel cache directory (default: ${XDG_CACHE_HOME:-$HOME/.cache}/inkling-ciru-triton)
  --original-weights      Use only the original checkpoint files, without fitted weights
  --dry-run               Print resolved environment and command without launching
EOF
}

die() {
    printf 'inkling-ciru: %s\n' "$*" >&2
    exit 2
}

mode=vision
model="${INKLING_MODEL_DIR:-}"
served_name=main
host=127.0.0.1
port=18753
context=
seqs=
batch_tokens=4096
prefill_chunk=
kv_mib=
spec_tokens=
spec_tokens_set=0
adaptive_spec=0
speculation=1
prefix_cache=auto
tools=0
vision=auto
turbo=1
no_graphs=0
draft_graph=auto
python_bin="${INKLING_PYTHON:-}"
overlay_dir="${INKLING_OVERLAY_DIR:-}"
transformers_src="${INKLING_TRANSFORMERS_SRC:-}"
transformers_deps=
triton_cache="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/inkling-ciru-triton}"
expert_bank=
expert_policy=source
coda_c1_dir=
original_weights=0
dry_run=0

while (( $# )); do
    case "$1" in
        --model|--mode|--served-model-name|--host|--port|--context|--seqs|--batch-tokens|--prefill-chunk-tokens|--kv-mib|--spec-tokens|--python|--overlay|--transformers-src|--transformers-deps|--triton-cache)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || die "missing value for $1"
            ;;
    esac
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --model) model="$2"; shift ;;
        --mode) mode="$2"; shift ;;
        --served-model-name) served_name="$2"; shift ;;
        --host) host="$2"; shift ;;
        --port) port="$2"; shift ;;
        --context) context="$2"; shift ;;
        --seqs) seqs="$2"; shift ;;
        --batch-tokens) batch_tokens="$2"; shift ;;
        --prefill-chunk-tokens) prefill_chunk="$2"; shift ;;
        --kv-mib) kv_mib="$2"; shift ;;
        --spec-tokens) spec_tokens="$2"; spec_tokens_set=1; shift ;;
        --adaptive-spec) adaptive_spec=1 ;;
        --no-adaptive-spec) adaptive_spec=-1 ;;
        --no-spec) speculation=0 ;;
        --prefix-cache) prefix_cache=1 ;;
        --no-prefix-cache) prefix_cache=0 ;;
        --tools) tools=1 ;;
        --vision) vision=1 ;;
        --language-model-only) vision=0 ;;
        --draft-graph) draft_graph=1 ;;
        --no-draft-graph) draft_graph=0 ;;
        --no-graphs) no_graphs=1 ;;
        --python) python_bin="$2"; shift ;;
        --overlay) overlay_dir="$2"; shift ;;
        --transformers-src) transformers_src="$2"; shift ;;
        --transformers-deps) transformers_deps="$2"; shift ;;
        --triton-cache) triton_cache="$2"; shift ;;
        --original-weights) original_weights=1 ;;
        --dry-run) dry_run=1 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

[[ -n "$model" ]] || die '--model PATH is required (download the release first)'
case "$mode" in
    single|long-context|concurrent|sampling|vision) ;;
    *) die "unsupported mode: $mode (expected vision, concurrent, single, long-context, or sampling)" ;;
esac
for number in "$port" "$batch_tokens" ${spec_tokens:+"$spec_tokens"} ${context:+"$context"} ${seqs:+"$seqs"} ${prefill_chunk:+"$prefill_chunk"} ${kv_mib:+"$kv_mib"}; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || die "expected a positive number, got $number"
done
(( port <= 65535 )) || die 'port must be at most 65535'
if (( spec_tokens_set )) && (( spec_tokens > 7 )); then
    die 'draft depth must be at most 7'
fi

context=${context:-1048576}
if [[ -z "$seqs" ]]; then
    if [[ "$mode" == concurrent || "$mode" == vision ]]; then seqs=8; else seqs=1; fi
fi
if [[ "$vision" == auto ]]; then
    if [[ "$mode" == vision ]]; then vision=1; else vision=0; fi
fi
if [[ "$prefix_cache" == auto ]]; then
    if [[ "$mode" == vision ]]; then prefix_cache=1; else prefix_cache=0; fi
fi
if (( ! original_weights )) && [[ -d "$model/optimized-weights" ]]; then
    expert_bank="$model/optimized-weights/experts"
    for layer in 6 12 17; do
        [[ -s "$expert_bank/layer$layer.pt" ]] || die "missing fitted expert weights for layer $layer; download the full bundle"
    done
    coda_c1_dir="$model/optimized-weights/final-layers"
    for layer in 20 21 22 23; do
        [[ -s "$coda_c1_dir/layer$layer-c1.safetensors" ]] || die "missing fitted final-layer weights for layer $layer; download the full bundle"
    done
    expert_policy=c1-down
fi

if [[ -z "$python_bin" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        python_bin="$VIRTUAL_ENV/bin/python"
    elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        python_bin="$REPO_ROOT/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        python_bin=$(command -v python3)
    else
        die 'could not locate python3; pass --python PATH'
    fi
fi

if [[ "$mode" == single || "$mode" == long-context ]] && (( seqs != 1 )); then
    die 'Use --mode vision or --mode concurrent for multiple sequences'
fi
if [[ -z "$kv_mib" ]]; then
    if (( context > 66048 )); then
        # Nine full-history BF16 cache visits with DSpark use 36 KiB per
        # token. Reserve an additional 16 GiB for windowed groups and slack.
        kv_mib=$((16384 + 36 * ((context + 1023) / 1024)))
    elif [[ "$mode" == long-context && "$context" == 66048 ]]; then
        kv_mib=28672
    elif (( seqs > 1 && context > 32768 )); then
        kv_mib=32768
    elif (( seqs > 1 && context >= 32768 )); then
        kv_mib=24576
    elif (( seqs > 1 && context > 4096 )); then
        kv_mib=10240
    else
        kv_mib=$((6144 * ((context + 16383) / 16384)))
    fi
fi

if (( adaptive_spec == 0 && turbo && speculation && !no_graphs && !spec_tokens_set && seqs > 1 )) && [[ "$mode" == concurrent || "$mode" == vision ]]; then
    adaptive_spec=1
fi
if (( adaptive_spec > 0 && seqs > 1 )); then
    spec_tokens=7
elif (( ! spec_tokens_set )); then
    if [[ "$mode" == sampling || "$mode" == concurrent || "$mode" == vision ]]; then
        spec_tokens=4
    else
        spec_tokens=7
    fi
fi

target_graph=0
plain_graph=0
copy_draft=0
verifier_head_tile=0
sample_method=greedy
experimental_kernels=0
draft_w8=0
router_mode=off
single_seq_dspark=0
greedy_fused_markov=0

case "$mode" in
    single|long-context)
        target_graph=$((speculation && !no_graphs))
        plain_graph=$((!speculation && !no_graphs))
        verifier_head_tile=$((8 * target_graph))
        single_seq_dspark="$target_graph"
        greedy_fused_markov="$target_graph"
        if (( turbo && speculation )); then
            experimental_kernels=1
            draft_w8=all
            router_mode=align
        fi
        ;;
    concurrent|vision)
        target_graph=$((speculation && !no_graphs))
        plain_graph=$((!speculation && !no_graphs))
        router_mode=align
        greedy_fused_markov="$target_graph"
        if (( turbo && target_graph )); then
            experimental_kernels=1
            draft_w8=all
        fi
        ;;
    sampling)
        sample_method=probabilistic
        target_graph=$((speculation && !no_graphs))
        plain_graph=$((!speculation && !no_graphs))
        router_mode=align
        ;;
esac
# The copy-draft optimization applies only to the short seven-draft graph.
if [[ "$mode" == single ]] && (( context <= 4096 && spec_tokens == 7 )); then
    copy_draft="$target_graph"
fi
if [[ "$draft_graph" == auto ]]; then
    if (( turbo && target_graph )); then
        draft_graph=1
    else
        draft_graph=0
    fi
fi

turbo_prefill_moe=0
if (( turbo )); then
    turbo_prefill_moe=fp16
fi

# Build clean PYTHONPATH (overlay first so patched vllm modules take precedence).
py_parts=()
if [[ -n "$overlay_dir" ]]; then
    py_parts+=("$overlay_dir")
fi
if [[ -n "$transformers_src" ]]; then
    py_parts+=("$transformers_src")
fi
if [[ -n "$transformers_deps" ]]; then
    py_parts+=("$transformers_deps")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    py_parts+=("$PYTHONPATH")
fi
joined_pypath=$(IFS=:; echo "${py_parts[*]}")

settings=(
    HF_HUB_OFFLINE=1
    VLLM_NO_USAGE_STATS=1
    VLLM_ENABLE_V1_MULTIPROCESSING=1
    VLLM_WORKER_MULTIPROC_METHOD=spawn
    VLLM_ROCM_USE_AITER=0
    "TRITON_CACHE_DIR=$triton_cache"
    OMP_NUM_THREADS=4
    OPENBLAS_NUM_THREADS=4
    "INKLING_AMD_EXPERT_POLICY=$expert_policy"
    "INKLING_AMD_EXPERT_BANK=$expert_bank"
    "INKLING_AMD_CODA_C1_DIR=$coda_c1_dir"
    INKLING_AMD_CODA_C1_KEEP_FINAL_MXFP8=0
    INKLING_AMD_PACKED_PREFILL=1
    INKLING_AMD_PACKED_MXFP8_PREFILL=1
    INKLING_AMD_PACKED_MXFP8=1
    INKLING_AMD_PACKED_LOOP=1
    INKLING_AMD_PACKED_ATTN_RES=1
    INKLING_AMD_QKVR_W8=0
    INKLING_AMD_PREFILL_WIDE_TILE=1
    INKLING_AMD_PREFILL_SHORT_N128=1
    INKLING_AMD_PREFILL_WIDE_SMALL_BATCH=1
    INKLING_AMD_PREFILL_FUSED_ACTIVATION=1
    INKLING_AMD_PREFILL_FUSED_REDUCE=1
    INKLING_AMD_PACKED_SMALL_M=0
    INKLING_AMD_GROUPWISE_W8_ATTN=1
    INKLING_AMD_W8_FUSED_QKV=1
    INKLING_AMD_SOURCE_M1_BF16_DOT2=1
    INKLING_AMD_SOURCE_M1_WAVE32=1
    INKLING_AMD_PACKED_M1_ROUTER=1
    INKLING_AMD_PACKED_M1_ROUTER_SCORES=1
    INKLING_AMD_PACKED_M1_ROUTER_SELECTION=1
    INKLING_AMD_M1_KEY_TILE=16
    INKLING_AMD_M1_ATTENTION_WAVES=2
    INKLING_AMD_M1_SPLIT_PROGRAMS_PER_CU=4
    INKLING_AMD_M1_COMPACT_SPLIT_MERGE=1
    INKLING_AMD_GROUPWISE_W6_ATTN=1
    INKLING_AMD_GROUPWISE_W8_HEAD=0
    INKLING_AMD_GROUPWISE_W6_HEAD=1
    INKLING_AMD_W6_HEAD_GROUP_SIZE=32
    INKLING_AMD_W6_HEAD_FUSED_SOFTCAP=1
    INKLING_AMD_W6_ATTN_WAVES=4
    INKLING_AMD_W6_HEAD_WAVES=4
    INKLING_AMD_LOOP_GEMV=both
    INKLING_AMD_DSPARK_PHASE_SWITCH=0
    INKLING_AMD_SOURCE_WMMA_N_FAST=0
    "INKLING_AMD_DECODE_GRAPH=$plain_graph"
    "INKLING_AMD_DSPARK_TARGET_GRAPH=$target_graph"
    "INKLING_AMD_SOURCE_WMMA_PRETILE=$((target_graph || turbo))"
    "INKLING_AMD_CODA_WMMA_PRETILE=$((target_graph || turbo))"
    "INKLING_AMD_MIXED_WMMA_PRETILE=$((target_graph || turbo))"
    "INKLING_AMD_SOURCE_WMMA_WAVE_EPILOGUE=$target_graph"
    "INKLING_AMD_DSPARK_COPY_DRAFT=$copy_draft"
    "INKLING_AMD_W6_VERIFY_ATTN=$target_graph"
    "INKLING_AMD_W6_HEAD_BATCH_TILE=$verifier_head_tile"
    "INKLING_AMD_DSPARK_W6_BASE_HEAD=$single_seq_dspark"
    "INKLING_AMD_DSPARK_W8_MARKOV=$single_seq_dspark"
    "INKLING_AMD_DSPARK_MARKOV_FUSED_ARGMAX=$greedy_fused_markov"
    "INKLING_AMD_DSPARK_W8_BACKBONE=$draft_w8"
    "INKLING_AMD_FUSED_M8_ROUTER=$router_mode"
    "INKLING_AMD_W6_BATCH_DOT2=$experimental_kernels"
    "INKLING_AMD_LOOP_M8_FUSED=$experimental_kernels"
    "INKLING_AMD_TURBO_PREFILL_MOE=$turbo_prefill_moe"
    "INKLING_AMD_TURBO_PREFILL_ATTN=$turbo"
    "INKLING_AMD_TURBO_MOE_F16=$turbo"
    "INKLING_AMD_TURBO_ROUTER_M8=$turbo"
    "INKLING_AMD_TURBO_W6_WMMA=$turbo"
    "INKLING_AMD_TURBO_DRAFT_W4=$turbo"
    INKLING_AMD_TURBO_MARKOV_TOPK=0
    "INKLING_AMD_DSPARK_DRAFT_GRAPH=$draft_graph"
)
bundled_hip_so="$SCRIPT_DIR/rocm/hip_inkling_gfx1151_671b45b80972.so"
if [[ -z "${INKLING_AMD_HIP_SO:-}" && -z "${INKLING_AMD_HIP_PREFILL_SO:-}" && -f "$bundled_hip_so" ]]; then
    settings+=("INKLING_AMD_HIP_SO=$bundled_hip_so")
fi
if [[ -n "$joined_pypath" ]]; then
    settings=("PYTHONPATH=$joined_pypath" "${settings[@]}")
fi
input_mode=()
if (( ! vision )); then
    input_mode+=(--language-model-only)
fi
command=(
    "$python_bin" -m vllm.entrypoints.cli.main serve "$model"
    --served-model-name "$served_name"
    --host "$host" --port "$port"
    "${input_mode[@]}" --dtype bfloat16 --reasoning-parser inkling
    --max-model-len "$context" --max-num-seqs "$seqs"
    --max-num-batched-tokens "$batch_tokens"
    --kv-cache-memory-bytes "$((kv_mib * 1048576))"
)
if [[ -n "$prefill_chunk" ]]; then
    command+=(--long-prefill-token-threshold "$prefill_chunk")
fi
if (( prefix_cache )); then
    command+=(--enable-prefix-caching)
else
    command+=(--disable-prefix-caching)
fi
cg_req_sizes=$(seq -s, 1 "$seqs")
if (( adaptive_spec > 0 && seqs > 1 )); then
    cg_max_toks=$((seqs * 5 > 8 ? seqs * 5 : 8))
else
    cg_max_toks=$((seqs * (spec_tokens + 1)))
fi
if (( target_graph )); then
    command+=(
        --compilation-config
        "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"decode_cudagraph_capture_request_sizes\":[$cg_req_sizes],\"max_decode_cudagraph_capture_tokens\":$cg_max_toks}"
        --cudagraph-metrics
    )
elif (( plain_graph )); then
    command+=(
        --compilation-config
        "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$cg_req_sizes]}"
        --cudagraph-metrics
    )
else
    command+=(--enforce-eager)
fi
if (( speculation )); then
    if (( adaptive_spec > 0 && seqs > 1 )); then
        command+=(
            --speculative-config
            "{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"num_speculative_tokens_per_batch_size\":[[1,1,7],[2,$seqs,4]],\"draft_sample_method\":\"$sample_method\"}"
        )
    else
        command+=(
            --speculative-config
            "{\"method\":\"dspark\",\"num_speculative_tokens\":$spec_tokens,\"draft_sample_method\":\"$sample_method\"}"
        )
    fi
fi
if (( tools )); then
    command+=(--enable-auto-tool-choice --tool-call-parser inkling)
fi
# Host selector supplies the native Inkling reasoning default.
if [[ -n "${INKLING_PROFILER_CONFIG:-}" ]]; then
    command+=(--profiler-config "$INKLING_PROFILER_CONFIG")
fi
if [[ -n "${INKLING_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    command+=(--default-chat-template-kwargs "$INKLING_CHAT_TEMPLATE_KWARGS")
fi
if (( dry_run )); then
    printf 'Inkling 2 Mini CIRU Halo Agent v1.5.0 (%s, adaptive=%s, vision=%s, prefix_cache=%s)\n' \
        "$mode" "$adaptive_spec" "$vision" "$prefix_cache"
    printf 'Environment:\n'
    printf '  %q\n' "${settings[@]}"
    printf 'Command:\n '
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

[[ -d "$model" ]] || die "model directory not found: $model"
[[ -x "$python_bin" ]] || die "python executable not found: $python_bin"
[[ -f "$SCRIPT_DIR/rocm/rocr-env.sh" ]] || die 'Packaged ROCm setup missing; re-run the installer'
[[ -f "$overlay_dir/vllm/models/inkling2/amd/model.py" ]] || die 'Use the launcher generated by scripts/install.sh'
[[ -f "$transformers_src/transformers/models/inkling2_loop/configuration_inkling2_loop.py" ]] || die 'Pinned Transformers source missing; re-run the installer'
[[ -d "$transformers_deps/huggingface_hub" && -d "$transformers_deps/safetensors" ]] ||
    die 'Pinned Transformers dependencies missing; follow INSTALL.md'
[[ -s "$model/config.json" && -s "$model/model-00001-of-00001.safetensors" ]] || die 'Checkpoint files missing from the model directory'
if (( speculation )); then
    [[ -s "$model/mtp.safetensors" ]] || die 'Bundled DSpark weights are missing from the model directory'
fi
[[ -r /dev/kfd && -w /dev/kfd ]] || die '/dev/kfd is not readable/writable; check render/video/kfd group permissions'
mkdir -p "$triton_cache"

source "$SCRIPT_DIR/rocm/rocr-env.sh"
ciru_use_fixed_rocr "$python_bin" || die 'ROCm idle CPU fix could not be loaded'

for s in "${settings[@]}"; do
    export "$s"
done
printf 'Starting Inkling 2 Mini CIRU Halo Agent v1.5.0, mode=%s context=%s sequences=%s KV memory=%s MiB adaptive=%s vision=%s\n' \
    "$mode" "$context" "$seqs" "$kv_mib" "$adaptive_spec" "$vision"
exec "${command[@]}"
