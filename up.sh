#!/usr/bin/env bash
# up.sh — bring the serve up on both boxes. Reads recipe.env and nothing else.
#
#   ./up.sh                      # serve, in the FOREGROUND
#   ./up.sh --dry-run            # print what would run, touch nothing
#
# 🔴 RUN THIS UNDER A SUPERVISOR, NOT UNDER A SHELL TIMEOUT.
#    Upstream's launch-cluster.sh runs in the foreground and installs
#    `trap cleanup EXIT INT TERM HUP`. A shell deadline kills the process GROUP and the
#    trap tears the cluster down — after the serve has already printed "Application
#    startup complete". `nohup` is not enough: it only ignores SIGHUP.
#
# 🔴 STEP 1 OF THIS SCRIPT IS DESTRUCTIVE BY DESIGN: it stops and removes the container
#    on BOTH boxes before doing anything else. That is required (a stale container on the
#    worker survives a clean head-side shutdown and keeps the weights resident), but it
#    means you cannot use this script to "test" anything while a serve is running.
#    Use --dry-run for that.
set -uo pipefail

ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0
case "${1:-}" in
  --dry-run) DRY=1 ;;
  "")        DRY=0 ;;
  *) printf '  ABORT: unknown argument "%s". usage: ./up.sh [--dry-run]\n' "$1" >&2; exit 2 ;;
esac

dire() { printf '  %s\n' "$*"; }
mourir() { printf '  ABORT: %s\n' "$*" >&2; exit 1; }

# ── config : ONE loader, shared by install.sh / up.sh / down.sh / view.sh.
# Four inlined copies of this had diverged; see _config.sh for what that cost.
# shellcheck disable=SC1090,SC1091
. "$ICI/_config.sh"
dire "config: recipe.env$([ -f "$ICI/recipe.env.local" ] && echo ' + recipe.env.local')"

for v in CLUSTER_NODES ETH_IF IB_IF MASTER_PORT CONTAINER_NAME IMAGE SERVED_DIR HF_HOME \
         CHECKPOINT_DIR KV_DTYPE GPU_MEM_UTIL MAX_LEN MAX_SEQS SPEC_TOKENS MOE_BACKEND; do
  [ -n "${!v:-}" ] || mourir "recipe.env does not set $v"
done

HEAD="${CLUSTER_NODES%%,*}"
WORKER="${CLUSTER_NODES##*,}"
[ "$HEAD" != "$WORKER" ] || mourir "CLUSTER_NODES must list two distinct hosts, head first"
SSH="ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no"
UP="$ICI/upstream"
[ -d "$UP" ] || mourir "upstream/ is missing — run ./install.sh first"

# ── must run ON the head. The head carries the HTTP server, the scheduler and the Ray
#    control plane, ~7 GiB more than the worker; running this from the worker silently
#    inverts the topology.
if ! ip -o addr show 2>/dev/null | grep -qw "$HEAD"; then
  mourir "this machine does not hold $HEAD (the head address). Run up.sh ON the head."
fi
$SSH "$WORKER" true 2>/dev/null || mourir "no key-based ssh from the head to $WORKER ($WORKER). \
The head drives the worker over ssh; it is not optional."

if [ "$DRY" = 1 ]; then
  dire "--dry-run: head=$HEAD worker=$WORKER image=$IMAGE model=$SERVED_DIR"
  dire "--dry-run: would stop/remove '$CONTAINER_NAME' on BOTH hosts, then launch."
fi

# ── 1. clean both boxes. `docker stop` FIRST: SIGKILL on a live CUDA process leaks
#    unified memory until the next reboot.
if [ "$DRY" = 0 ]; then
  for h in "" "$WORKER"; do
    C="docker stop -t 20 $CONTAINER_NAME >/dev/null 2>&1; docker rm -f $CONTAINER_NAME >/dev/null 2>&1"
    if [ -z "$h" ]; then eval "$C"; else $SSH "$h" "$C" 2>/dev/null; fi
  done
  dire "container '$CONTAINER_NAME' removed on both hosts"

  # ── 2. give the checkpoint's page cache back. vLLM's startup guard reads MemFree, NOT
  #    MemAvailable: on unified memory the page cache counts AGAINST you, and a
  #    checkpoint you just read can alone make a fitting serve refuse to boot.
  #    ⚠️ This step runs the SAME script on the worker, at the SAME absolute path. Until
  #    2026-09-07 it did so blind (`>/dev/null`), so on any box where the repo was cloned only
  #    on the head it silently did nothing there — a load-bearing step failing without a word,
  #    which this repo forbids everywhere else. It now refuses, and says why.
  $SSH "$WORKER" "test -f '$ICI/tools/free-checkpoint-cache.py'" \
    || mourir "the worker has no copy of this repo at $ICI.
       Clone it there at the SAME path (README, Requirements), then rerun."
  for h in "" "$WORKER"; do
    C="CHECKPOINT_DIR='$CHECKPOINT_DIR' python3 '$ICI/tools/free-checkpoint-cache.py'"
    if [ -z "$h" ]; then OUT=$(eval "$C" 2>&1); RC=$?; else OUT=$($SSH "$h" "$C" 2>&1); RC=$?; fi
    if [ "$RC" -eq 0 ]; then dire "page cache released on ${h:-head}: $(printf '%s' "$OUT" | tail -1)"
    else dire "WARN: could not release the checkpoint's page cache on ${h:-head} (rc=$RC): $(printf '%s' "$OUT" | tail -1)"
         dire "      the boot may refuse on MemFree; see README, Requirements."; fi
  done

  # ── 3. orphan shared-memory segments. A crashed boot leaves root-owned
  #    /dev/shm/psm_* that a normal user cannot delete, so free memory RATCHETS DOWN
  #    across retries. It can only be reported — lowering gmu never catches up, only a
  #    reboot does.
  ORPH=0
  for f in /dev/shm/psm_*; do
    [ -e "$f" ] || continue
    grep -lsq "$(basename "$f")" /proc/[0-9]*/maps 2>/dev/null && continue
    rm -f "$f" 2>/dev/null || ORPH=$((ORPH+1))
  done
  [ "$ORPH" -gt 0 ] && dire "WARN: $ORPH root-owned /dev/shm/psm_* remain (the ratchet). If the guard below trips, REBOOT."
fi

# ── 4. the only number that decides whether the boot can succeed.
#    🔴 THIS GUARD USED TO FAIL OPEN and it is the single decision protecting a
#    swapless unified-memory box. `set -uo pipefail` has no `-e`: when the probe below
#    failed, FREE/TOTAL were EMPTY, the arithmetic raised, the `if` returned non-zero,
#    the ABORT branch was NOT taken — and the boot proceeded blind. Any non-numeric
#    reading is now refused explicitly.
# ⚠️ --dry-run must touch NOTHING, so the probe itself is guarded. It is `--rm` and
#    read-only, but "harmless" is a claim and a guard is a fact.
if [ "$DRY" = 1 ]; then
  dire "--dry-run: skipping the CUDA memory probe (it would run a container)"
  FREE=999; TOTAL=121.69
else
  read -r FREE TOTAL < <(docker run --rm --gpus all --ipc=host "$IMAGE:latest" python3 -c \
    'import torch; f,t=torch.cuda.mem_get_info(); print(f"{f/(1<<30):.2f} {t/(1<<30):.2f}")' 2>/dev/null | tail -1)
  case "${FREE:-}${TOTAL:-}" in *[!0-9.]*|"") FREE=""; TOTAL="" ;; esac
  if [ -z "${FREE:-}" ] || [ -z "${TOTAL:-}" ]; then
    mourir "the memory probe returned nothing (FREE='${FREE:-}' TOTAL='${TOTAL:-}').
       An UNOBSERVED state is not a healthy state: refusing to boot blind.
       Check: image '$IMAGE:latest' exists, you are in the docker group, the GPU is free."
  fi
fi
NEED=$(python3 -c "print(f'{$GPU_MEM_UTIL*$TOTAL:.2f}')") || mourir "cannot compute the requirement"
dire "CUDA free ${FREE} / ${TOTAL} GiB   gmu ${GPU_MEM_UTIL} needs ${NEED} GiB"
# container start + Ray init eat ~2.5 GiB between this probe and the allocation
if python3 -c "import sys; sys.exit(0 if $FREE - 2.5 < $NEED else 1)"; then
  [ "$DRY" = 1 ] && dire "--dry-run: the guard WOULD abort here" || mourir \
    "not enough free memory. Stop background I/O, or reboot (root-owned psm_* cannot be
       cleared otherwise). Safe gmu <= $(python3 -c "print(f'{($FREE-2.5)/$TOTAL:.3f}')")"
fi

# ── 5. render upstream's env file from recipe.env.
#    Upstream reads CONTAINER_* from a .env and turns each into `-e NAME=value`. That
#    file is in ITS .gitignore, so a fresh clone of upstream has NO interconnect
#    configuration and NO sm_121 correctness flags: NCCL then falls back to TCP on the
#    same cable, silently, and int4 MoE returns garbage. recipe.env is the single source;
#    the file it expects is rendered and passed with --config.
RENDU="$(mktemp -d)/recipe.env.rendered"
{
  echo "# generated by up.sh from recipe.env$([ -f "$ICI/recipe.env.local" ] && echo ' + recipe.env.local') — do not edit"
  echo "CLUSTER_NODES=$CLUSTER_NODES"
  echo "ETH_IF=$ETH_IF"
  echo "IB_IF=$IB_IF"
  echo "MASTER_PORT=$MASTER_PORT"
  echo "CONTAINER_NAME=$CONTAINER_NAME"
  env | grep '^CONTAINER_' | grep -v '^CONTAINER_NAME=' | sort
} > "$RENDU"
NB=$(grep -c '^CONTAINER_' "$RENDU")
dire "rendered $NB CONTAINER_* variables for upstream --config"
[ "$NB" -ge 8 ] || mourir "only $NB CONTAINER_* variables rendered; recipe.env looks truncated.
       Without them NCCL runs over TCP and int4 MoE returns garbage, both silently."

# ── 6. the passthrough whitelist.
#    ⚠️ THIS LIST IS A FILTER, NOT DOCUMENTATION. A variable missing here never reaches
#    the container and the profile keeps its default IN SILENCE — verified upstream:
#    a flag set on the command line, a successful boot, and the opposite value in the
#    engine config, with no message anywhere. Every variable a profile reads MUST be
#    added here at the same time.
#    The list below is exactly what overlay/examples/q38.sh reads, plus the VLLM_* the
#    engine and the mods read directly.
# 🔴 THE MODEL PATH THE ENGINE SEES IS A CONTAINER PATH, NOT A HOST PATH.
#    Upstream mounts `-v $HF_HOME:/root/.cache/huggingface` (launch-cluster.sh:8), plus the
#    ~/.cache/vllm, ~/.cache/flashinfer and ~/.triton caches. Passing the host path makes
#    vLLM look for the checkpoint at a location that does not exist inside the container.
#    Verified on the live container:
#      <HF_HOME on the host> -> /root/.cache/huggingface
#    Therefore SERVED_DIR must live UNDER HF_HOME, and it is translated. If it does not,
#    the boot is refused: an unmounted checkpoint is a 10-minute boot that dies at the end.
case "$SERVED_DIR" in
  "$HF_HOME"/*) MODEL_IN_CONTAINER="/root/.cache/huggingface/${SERVED_DIR#"$HF_HOME"/}" ;;
  /root/.cache/huggingface/*) MODEL_IN_CONTAINER="$SERVED_DIR" ;;   # already container-side
  *) mourir "SERVED_DIR ($SERVED_DIR) is not under HF_HOME ($HF_HOME).
       Only HF_HOME is mounted into the container, so the engine could not open it.
       Move the served directory under HF_HOME, or set HF_HOME to its parent." ;;
esac
dire "model (in container): $MODEL_IN_CONTAINER"
ENVS=(-e "MODEL_PATH=$MODEL_IN_CONTAINER" -e "GPU_MEM_UTIL=$GPU_MEM_UTIL" -e "KV_DTYPE=$KV_DTYPE" -e "MAX_LEN=$MAX_LEN")
for v in HOST PORT MOE_BACKEND EXTRA_ARGS SPEC_TOKENS SPEC_MODEL BATCHED_TOKENS LONG_PREFILL MAX_SEQS \
         KV_BYTES CG CG_SIZES EAGER EP ALL2ALL IDX_SHARE MAMBA_DTYPE ROPE_FACTOR \
         VLLM_QSA_KV_FP8 VLLM_QWEN4EXP_PLE_FP8 VLLM_DEEP_GEMM_WARMUP VLLM_LOGGING_LEVEL; do
  [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}")
done

# ── 7. the mods. Order does not matter; each refuses to apply twice.
MODS_DEF="mods/opt-sm121-gemv mods/ple-fp8-mixed mods/mtp-modelopt-mixed"
if [ "${KV_DTYPE}" = "fp8_e4m3" ]; then
  MODS_DEF="mods/qsa-kv-fp8 $MODS_DEF"
elif [ -n "${VLLM_QSA_KV_FP8:-}" ] && [ "${VLLM_QSA_KV_FP8}" != "0" ]; then
  mourir "VLLM_QSA_KV_FP8=$VLLM_QSA_KV_FP8 but KV_DTYPE=$KV_DTYPE.
       Those two must agree: the flag alone patches nothing, and fp8 without the mod is
       refused by the engine's own guard. Set both, or neither."
fi
MODS="${MODS:-$MODS_DEF}"
MODARGS=(); for m in $MODS; do MODARGS+=(--apply-mod "$m"); done

dire "launch: model=$SERVED_DIR kv=$KV_DTYPE moe=$MOE_BACKEND gmu=$GPU_MEM_UTIL"
dire "        mods=$MODS"
if [ "$DRY" = 1 ]; then
  dire "--dry-run: would exec, in $UP:"
  printf '    ./launch-cluster.sh -t %s --config %s %s %s --launch-script examples/q38.sh\n' \
    "$IMAGE" "$RENDU" "${ENVS[*]}" "${MODARGS[*]}"
  exit 0
fi
cd "$UP" || mourir "cannot enter $UP"
exec env HF_HOME="$HF_HOME" \
  ./launch-cluster.sh -t "$IMAGE" --config "$RENDU" "${ENVS[@]}" "${MODARGS[@]}" \
  --launch-script examples/q38.sh
