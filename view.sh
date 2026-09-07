#!/usr/bin/env bash
# view.sh — the four things worth checking on a live serve, and nothing else.
#
# Each block exists because the obvious check is wrong somewhere:
#   1. "Application startup complete" can print AFTER an EngineDeadError. /health alone
#      is not enough either — you need both, plus the absence of the error in the log.
#   2. A flag in the container's environment is NOT a flag the engine consumed. And
#      `docker inspect --format '{{.Config.Cmd}}'` is USELESS with this launcher: it
#      returns `sleep infinity`, because vLLM is started with `docker exec`. The only
#      valid witness is the engine's own `non-default args` echo.
#   3. NCCL happily runs over TCP on the same ConnectX cable and never says so: the
#      serve works and every step is ~2x slower. The witness is the HCA byte counter
#      moving while the interface's TCP counter stays flat.
#   4. On unified memory MemAvailable counts reclaimable page cache; the CUDA driver
#      wants pages that are FREE. Both numbers, on both ranks, or neither.
set -uo pipefail

ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$ICI/_config.sh"
HEAD="${CLUSTER_NODES%%,*}"; WORKER="${CLUSTER_NODES##*,}"
SSH="ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no"
CN="${CONTAINER_NAME:-vllm_node}"
h1() { printf '\n── %s\n' "$*"; }

h1 "1. alive?"
printf '   /health           %s\n' "$(curl -s -o /dev/null -w '%{http_code}' -m 15 "http://${HOST:-127.0.0.1}:${PORT:-8000}/health")"
printf '   container         %s\n' "$(docker ps --format '{{.Names}}={{.Status}}' 2>/dev/null | grep "^$CN=" || echo 'ABSENT')"
printf '   worker container  %s\n' "$($SSH "$WORKER" "docker ps --format '{{.Names}}={{.Status}}'" 2>/dev/null | grep "^$CN=" || echo 'ABSENT or unobservable')"

h1 "2. what the ENGINE actually consumed (not what you set)"
# Source: the live `vllm serve` process itself — its argv and its environment. The engine's
# "non-default args" echo goes to the launcher's stdout (your supervisor log), not to
# `docker logs` nor to Ray's logs, so it is not readable from here.
PID=$(docker exec "$CN" pgrep -of 'vllm serve' 2>/dev/null)
if [ -z "$PID" ]; then
  printf '   no `vllm serve` process in %s — the engine is not running\n' "$CN"
else
  # flags come in both forms: `--flag value` (two argv) and `--flag=value` (one argv)
  docker exec "$CN" cat "/proc/$PID/cmdline" 2>/dev/null | tr '\0' '\n' | awk '
    /^--(served-model-name|moe-backend|kv-cache-dtype|kv-cache-memory-bytes|speculative-config|prefix-match-unit|max-num-batched-tokens|max-model-len|gpu-memory-utilization|host|port|attention-backend)(=|$)/ {
      if ($0 ~ /=/) { print "   " $0 } else { flag=$0; getline; print "   " flag "=" $0 } }
    /^[^-]/ && prev_is_serve { print "   model=" $0 } { prev_is_serve = ($0 == "serve") }'
  docker exec "$CN" cat "/proc/$PID/environ" 2>/dev/null | tr '\0' '\n' \
    | grep -E '^(VLLM_QSA_KV_FP8|VLLM_QWEN4EXP_PLE_FP8|VLLM_USE_FLASHINFER_MOE_INT4|NCCL_IB_HCA|NCCL_NET_GDR_DISABLE)=' | sed 's/^/   env /'
  printf '   (argv and environment of pid %s; the "non-default args" echo is in the supervisor log)\n' "$PID"
fi

h1 "3. RDMA or silent TCP fallback?"
C="/sys/class/infiniband/${IB_IF}/ports/1/counters/port_xmit_data"
if [ -r "$C" ]; then
  R0=$(cat "$C"); T0=$(awk -v i="$ETH_IF" '$1 ~ i {print $10}' /proc/net/dev)
  curl -s -m 120 -o /dev/null "http://${HOST:-127.0.0.1}:${PORT:-8000}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"q38","messages":[{"role":"user","content":"Count from 1 to 60, one per line."}],"max_tokens":400,"temperature":0}'
  R1=$(cat "$C"); T1=$(awk -v i="$ETH_IF" '$1 ~ i {print $10}' /proc/net/dev)
  python3 - "$R0" "$R1" "$T0" "$T1" <<'PY'
import sys
r0, r1, t0, t1 = (int(x) for x in sys.argv[1:5])
rdma = (r1 - r0) * 4 / 1e6      # the counter is in 4-byte words
tcp = (t1 - t0) / 1e6
print(f"   RDMA  +{rdma:9.1f} MB")
print(f"   TCP   +{tcp:9.1f} MB   on the same link")
if rdma > 0 and rdma > tcp * 10:
    print("   [OK] RDMA is carrying the collectives.")
elif rdma == 0 and tcp > 0:
    print("   FALLBACK: NCCL is on TCP sockets. Check /dev/infiniband in the container,")
    print("             CONTAINER_NCCL_IB_* in recipe.env, and `ulimit -l`.")
else:
    print("   INCONCLUSIVE: nothing crossed the link. Was the serve generating?")
PY
else
  printf '   %s unreadable — IB_IF wrong, or rdma-core not installed.\n' "$C"
fi

h1 "4. memory, both ranks (MemFree is the one the CUDA driver cares about)"
for h in "" "$WORKER"; do
  N="head  "; [ -n "$h" ] && N="worker"
  C2='awk "/^MemFree|^MemAvailable/{printf \"%s %.2f  \", \$1, \$2/1048576}" /proc/meminfo'
  if [ -z "$h" ]; then O=$(eval "$C2" 2>/dev/null); else O=$($SSH "$h" "$C2" 2>/dev/null); fi
  printf '   %s  %s\n' "$N" "${O:-unobservable}"
done
printf '\n   Low MemFree with high MemAvailable = checkpoint page cache.\n'
printf '   CHECKPOINT_DIR=%s python3 tools/free-checkpoint-cache.py   # returns it, on each rank\n' "${CHECKPOINT_DIR:-...}"
