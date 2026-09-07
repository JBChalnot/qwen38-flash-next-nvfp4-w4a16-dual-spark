#!/usr/bin/env bash
# down.sh — stop the serve on BOTH boxes, and PROVE it stopped.
#
# 🔴 Stopping a two-node serve is a STATE TO VERIFY, not an event. Upstream's teardown is
#    asymmetric: it stops the head rank and then dies, leaving the worker alive with the
#    weights resident (observed: worker at 0.96 GiB free while the head looked clean).
#    A zero exit code on the head says nothing about the worker.
#
# An UNREACHABLE node is not a CLEAN node. A rank that cannot be observed is reported
#    "unobservable" with a non-zero exit — never as a success that was not seen.
set -uo pipefail

ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$ICI/_config.sh"

HEAD="${CLUSTER_NODES%%,*}"; WORKER="${CLUSTER_NODES##*,}"
SSH="ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no"
CN="${CONTAINER_NAME:-vllm_node}"
dire() { printf '  %s\n' "$*"; }

# `docker stop` BEFORE `docker rm -f`: a SIGKILL on a live CUDA process leaks unified
# memory until the next reboot, and no amount of retrying gets it back.
for h in "" "$WORKER"; do
  C="docker stop -t 20 $CN >/dev/null 2>&1; docker rm -f $CN >/dev/null 2>&1"
  if [ -z "$h" ]; then eval "$C"; else $SSH "$h" "$C" 2>/dev/null; fi
done

ECHEC=0
for h in "" "$WORKER"; do
  NOM="head($HEAD)"; [ -n "$h" ] && NOM="worker($h)"
  # `docker ps` must be OBSERVED to have run: its own exit code is captured before the pipe,
  # otherwise a missing docker or a denied socket prints "0" and reads as "clean".
  C="L=\$(docker ps --format '{{.Names}}' 2>/dev/null); D=\$?; [ \"\$D\" -eq 0 ] || exit 9; printf '%s\\n' \"\$L\" | grep -cx '$CN'; awk '/^MemFree/{printf \" %.2f\", \$2/1048576}' /proc/meminfo"
  if [ -z "$h" ]; then OUT=$(eval "$C" 2>/dev/null); RC=$?
  else OUT=$($SSH "$h" "$C" 2>/dev/null); RC=$?; fi
  if [ "$RC" -ne 0 ] || [ -z "$OUT" ]; then
    dire "🔴 $NOM: UNOBSERVABLE (probe rc=$RC). Not 'clean' — unknown."
    ECHEC=1; continue
  fi
  RESTE=$(printf '%s\n' "$OUT" | head -1)
  LIBRE=$(printf '%s\n' "$OUT" | tail -1 | tr -d ' ')
  if [ "$RESTE" != "0" ]; then
    dire "🔴 $NOM: '$CN' STILL RUNNING after stop+rm."
    ECHEC=1
  else
    dire "[OK] $NOM: no '$CN' · MemFree ${LIBRE} GiB"
  fi
done

# Unified memory takes 30-60 s to come back after a container dies. Launching before it
# does gives a phantom "CUDA out of memory" that looks like a configuration error and is not one.
[ "$ECHEC" -eq 0 ] && dire "STOP VERIFIED on BOTH ranks. Wait ~30-60 s before booting again:" \
  && dire "unified memory is not returned instantly, and an early relaunch reads a phantom OOM."
exit "$ECHEC"
