#!/usr/bin/env bash
# install.sh — build the engine image this recipe was measured on, from source, at pinned
#              commits, and prove the result is that engine.
#
#   ./install.sh --check     # read-only: prerequisites and, if the image exists, its pins
#   ./install.sh             # clone + build (skipped if the image already holds the pin) + verify
#   ./install.sh --rebuild   # build even if an image at the pin exists
#
# The image is built in THREE stages, none of them upstream's own build path:
#   1. build/base-arm64   : a manylinux-like arm64 build base (vLLM's Dockerfile expects
#                           pytorch/manylinux2_28-builder, which has no arm64 manifest)
#   2. vLLM's own docker/Dockerfile at $VLLM_REF, target vllm-openai, with that base
#   3. build/workspace    : /workspace + ray + empty entrypoint, for the cluster launcher
# eugr/spark-vllm-docker is cloned for ONE thing: launch-cluster.sh, the two-node launcher.
# Its Dockerfile and build-and-copy.sh are not used; they can download prebuilt wheels from
# moving tags, which would give a different engine while the patch scripts still apply.
set -uo pipefail

ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UP="$ICI/upstream"
VL="$ICI/upstream-vllm"
PINS="$ICI/.pins"

# ── the pins. Changing any of these invalidates the measurements. See PROVENANCE.md.
EUGR_REPO="https://github.com/eugr/spark-vllm-docker"
EUGR_REF="30f72c8a36e3720d3c8785fa36884eab8196d45c"
VLLM_REPO="https://github.com/vllm-project/vllm.git"
VLLM_REF="f561eca6ca4f3f79808a696b1521cb76dc8aafa2"     # PR #53899 branch; fetchable by sha
BASE_IMAGE="vllm-build-base-arm64:cuda13.0.3"
RAY_ATTENDU="2.56.1"
TORCH_ATTENDU="2.13.0+cu130"
FLASHINFER_ATTENDU="0.6.17"                              # pinned inside vLLM's Dockerfile at $VLLM_REF

# ⚠️ REJETER un argument inconnu, pas l'ignorer. La v1 traitait tout ce qui n'etait pas
# `--check` comme « construis », donc une faute de frappe (ou un `--dry-run` par reflexe)
# lancait un build de plusieurs heures.
CHECK=0; REBUILD=0
[ $# -le 1 ] || { printf '  FAIL CLOSED: one argument at most (got %s).\n' "$#" >&2; exit 2; }
case "${1:-}" in
  --check)   CHECK=1 ;;
  --rebuild) REBUILD=1 ;;
  "")        CHECK=0 ;;
  *) printf '  FAIL CLOSED: unknown argument "%s".\n' "$1" >&2
     printf '  usage: ./install.sh [--check|--rebuild]\n' >&2
     printf '    (no argument) clone, build and verify -- takes hours\n' >&2
     printf '    --check       read-only: report what it would do, build nothing\n' >&2
     printf '    --rebuild     build even if an image at the pinned commit already exists\n' >&2
     printf '  There is no --dry-run here: --check IS the dry run.\n' >&2
     exit 2 ;;
esac
dire() { printf '  %s\n' "$*"; }
mourir() { printf '  FAIL CLOSED: %s\n' "$*" >&2; exit 1; }
ok() { printf '  [OK] %s\n' "$*"; }

verifier_moteur_si_possible() {
  dire "── verifying the built engine"
  V=$(docker run --rm --entrypoint python3 "$IMAGE:latest" -c \
      'import vllm;print(vllm.__version__)' 2>/dev/null)
  [ -n "$V" ] || mourir "cannot read vllm.__version__ from $IMAGE:latest"
  case "$V" in
    *"${VLLM_REF:0:9}"*) ok "vllm $V contains the pinned ${VLLM_REF:0:9}" ;;
    *) mourir "vllm is $V, which does NOT contain ${VLLM_REF:0:9}.
       You most likely got the PREBUILT WHEEL path instead of a source build. Every
       number in README.md was measured on ${VLLM_REF:0:9}; on another engine they do not
       apply, and the patch scripts may still apply cleanly, which is worse." ;;
  esac
  for p in ray:"$RAY_ATTENDU" torch:"$TORCH_ATTENDU" flashinfer-python:"$FLASHINFER_ATTENDU"; do
    N=${p%%:*}; A=${p##*:}
    G=$(docker run --rm --entrypoint python3 "$IMAGE:latest" -c \
        "import importlib.metadata as m;print(m.version('$N'))" 2>/dev/null)
    [ "$G" = "$A" ] && ok "$N $G" || dire "WARN: $N is $G, measured on $A.
      TP=2 runs over Ray; a minor bump changes the executor. Upstream does not pin it."
  done
}

derive_des_ancres() {
  # ⚠️ CE CONTROLE DOIT TOURNER DANS LE CONTENEUR, PAS SUR L'HOTE.
  # La v1 extrayait l'arbre vllm dans un temporaire de l'HOTE et y lancait les scripts avec
  # le python de l'hote. Deux consequences, trouvees le 2026-09-07 :
  #  (a) `opt-sm121-gemv` verifie son module en l'important, donc il a besoin de `torch` --
  #      absent de l'hote chez la plupart des gens => refus sur un chemin SAIN (
  #      un etat non observe n'est pas un etat sain) ;
  #  (b) plus grave, ca ne mesurait pas le regime servi. En production ces scripts tournent
  #      DANS le conteneur, avec son python et ses paquets. Un controle qui n'a pas le meme
  #      environnement que la chose controlee ne controle rien .
  # La copie reste jetable : l'arbre est duplique dans /tmp du conteneur, le conteneur est
  # `--rm`, et l'image n'est jamais ecrite.
  dire "── patch scripts against the built tree (inside the container, as in production)"
  ECHECS=0
  for m in "$ICI"/overlay/mods/*/; do
    N=$(basename "$m")
    if SORTIE=$(docker run --rm --entrypoint bash \
         -e VLLM_QSA_KV_FP8=1 \
         -v "$m":/mod:ro "$IMAGE:latest" -c '
           set -e
           V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
           T=$(mktemp -d); cp -a "$V/." "$T/"
           VLLM_DIR="$T" bash /mod/run.sh
         ' 2>&1); then
      # PROVENANCE.md promet que --check « rapporte ses comptes de sites sur VOTRE arbre ».
      # La v1 jetait la sortie dans /dev/null, donc la promesse etait fausse. On la montre.
      ok "$N applies"
      printf '%s\n' "$SORTIE" | sed 's/^/       /'
    else
      dire "🔴 $N REFUSES to apply on this engine — its anchors moved."
      dire "   re-run it verbosely to see which one:"
      dire "   docker run --rm --entrypoint bash -v $m:/mod:ro $IMAGE:latest -c '"
      dire "     V=\$(python3 -c \"import vllm,os;print(os.path.dirname(vllm.__file__))\")"
      dire "     T=\$(mktemp -d); cp -a \"\$V/.\" \"\$T/\"; VLLM_DIR=\$T bash /mod/run.sh'"
      ECHECS=$((ECHECS+1))
    fi
  done
  [ "$ECHECS" -eq 0 ] || mourir "$ECHECS patch script(s) refuse to apply. Do NOT serve: read
         PROVENANCE.md, and re-read the anchors against this vLLM revision."
}

. "$ICI/_config.sh"

# ── 0. prerequisites, each one a reported failure mode
dire "── prerequisites"
for b in docker git python3 ssh curl; do
  command -v "$b" >/dev/null || mourir "$b is not on PATH"
done
docker info >/dev/null 2>&1 || mourir "cannot talk to docker. Are you in the docker group?
       (\`sudo usermod -aG docker \$USER\`, then log out and back in)"
ok "docker reachable, tools present"

HEAD="${CLUSTER_NODES%%,*}"; WORKER="${CLUSTER_NODES##*,}"
[ "$HEAD" != "$WORKER" ] || mourir "CLUSTER_NODES must list two distinct hosts, head first"
ip -o addr show 2>/dev/null | grep -qw "$HEAD" || mourir \
  "this machine does not hold $HEAD. Run install.sh ON the head rank."
ssh -o ConnectTimeout=5 -o BatchMode=yes "$WORKER" true 2>/dev/null || mourir \
  "no key-based ssh from the head to the worker ($WORKER).
       The head drives the worker over ssh for the whole lifetime of the serve.
       \`ssh-copy-id $WORKER\` once, then re-run."
ok "ssh head -> worker works without a password"

# ── the RDMA prerequisites. Not fatal: the recipe runs over TCP too, ~2x slower.
if [ -d /dev/infiniband ]; then
  ok "/dev/infiniband present on the head"
  ssh -o BatchMode=yes "$WORKER" 'test -d /dev/infiniband' 2>/dev/null \
    && ok "/dev/infiniband present on the worker" \
    || dire "WARN: no /dev/infiniband on the worker — NCCL will use TCP, ~2x slower."
else
  dire "WARN: no /dev/infiniband here. Install rdma-core, or accept the TCP fallback."
fi
[ -r "/sys/class/infiniband/${IB_IF:-none}/ports/1/counters/port_xmit_data" ] \
  && ok "IB_IF=$IB_IF resolves to a real HCA" \
  || dire "WARN: IB_IF='${IB_IF:-}' does not resolve. ./view.sh will tell you if RDMA is live."

# ── memory. The guard in up.sh needs gmu*total free, and page cache counts AGAINST you.
for h in "" "$WORKER"; do
  N="head"; [ -n "$h" ] && N="worker"
  C='awk "/^MemFree/{printf \"%.1f\", \$2/1048576}" /proc/meminfo'
  if [ -z "$h" ]; then M=$(eval "$C"); else M=$(ssh -o BatchMode=yes "$h" "$C" 2>/dev/null); fi
  [ -n "$M" ] || { dire "WARN: cannot read MemFree on $N"; continue; }
  dire "$N MemFree ${M} GiB (a cold boot of this recipe wants >= 104)"
done

# ── disk. The checkpoint is 123.6 GiB and it must exist on BOTH ranks.
if [ -n "${CHECKPOINT_DIR:-}" ] && [ -d "$CHECKPOINT_DIR" ]; then
  T=$(du -sb "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
  if [ "${T:-0}" -eq 132734506208 ]; then
    ok "checkpoint byte-exact on the head (132,734,506,208 B)"
  else
    dire "WARN: checkpoint is ${T:-0} B, expected 132,734,506,208 — wrong revision or incomplete."
    dire "      Pin it: huggingface-cli download nvidia/Qwen3.8-Flash-Next-NVFP4 \\"
    dire "              --revision fab0aecb760cec45227f6656abcaafa11abca87a"
  fi
  R=$(ssh -o BatchMode=yes "$WORKER" "du -sb '$CHECKPOINT_DIR' 2>/dev/null | cut -f1" 2>/dev/null)
  [ "${R:-0}" -eq 132734506208 ] && ok "checkpoint byte-exact on the worker" \
    || dire "WARN: worker checkpoint is ${R:-absent} B. TP=2 needs it on BOTH ranks."
else
  dire "WARN: CHECKPOINT_DIR ('${CHECKPOINT_DIR:-}') is not a directory yet."
fi

if [ "$CHECK" = 1 ]; then
  dire ""
  dire "--check: nothing was cloned, built or modified."
  if docker image inspect "$IMAGE:latest" >/dev/null 2>&1; then
    verifier_moteur_si_possible
    # ⚠️ ET LE CONTROLE ANTI-DERIVE, qui est la raison d'etre de --check.
    # PROVENANCE.md dit « ./install.sh --check reports the anchor counts » ; la v1 sortait
    # AVANT de le faire, donc la promesse etait fausse. Corrige le 2026-09-07. C'est en
    # lecture seule : l'arbre vllm est extrait de l'image dans un temporaire, les scripts
    # patchent LA COPIE, et le temporaire est detruit. L'image n'est jamais touchee.
    derive_des_ancres
  else
    dire "--check: image $IMAGE:latest not built yet, engine and anchor checks skipped"
  fi
  exit 0
fi

# ── 1. the cluster launcher: eugr/spark-vllm-docker at the pinned commit (launch-cluster.sh only).
dire "── upstream launcher tree"
if [ -d "$UP/.git" ]; then
  A=$(git -C "$UP" rev-parse HEAD 2>/dev/null)
  if [ "$A" = "$EUGR_REF" ]; then
    ok "upstream/ already at $EUGR_REF"
  elif [ -z "$(git -C "$UP" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    # a clean clone at another commit (an interrupted earlier run, or a stale checkout):
    # nothing to lose, move it to the pin. Local edits are never discarded.
    git -C "$UP" fetch -q origin 2>/dev/null
    git -C "$UP" checkout -q "$EUGR_REF" || mourir "upstream/ is at $A and $EUGR_REF cannot be checked out.
       Delete upstream/ and re-run."
    ok "upstream/ moved from $A to $EUGR_REF (clean tree, nothing discarded)"
  else
    mourir "upstream/ is at $A with LOCAL EDITS, expected $EUGR_REF.
       Commit or discard them, or delete upstream/ and re-run."
  fi
else
  git clone -q "$EUGR_REPO" "$UP" || mourir "clone failed"
  if ! git -C "$UP" checkout -q "$EUGR_REF"; then
    rm -rf "$UP"   # do not leave a half-state that blocks the next run
    mourir "checkout $EUGR_REF failed; upstream/ removed. Is the pin a public commit?"
  fi
  ok "cloned $EUGR_REPO at $EUGR_REF"
fi

{ echo "# resolved at install time, $(date -u +%FT%TZ)"
  echo "eugr=$EUGR_REF"; echo "vllm=$VLLM_REF"; echo "base_image=$BASE_IMAGE"
} > "$PINS" || mourir "cannot write $PINS"

# ── 2. the overlay: the profile and the mods, on top of the upstream tree.
dire "── overlay"
mkdir -p "$UP/examples" "$UP/mods" || mourir "cannot create $UP/{examples,mods}"
cp -r "$ICI/overlay/examples/." "$UP/examples/" || mourir "cannot copy the profile into upstream/"
cp -r "$ICI/overlay/mods/." "$UP/mods/" || mourir "cannot copy the mods into upstream/"
ok "profile and $(find "$ICI/overlay/mods" -maxdepth 1 -mindepth 1 -type d | wc -l) mods copied into upstream/"

# ── 3. build, three stages. Skipped when $IMAGE:latest already exists AND contains the
#    pinned vLLM commit; `--rebuild` forces. An image at another commit is refused, not reused.
if [ "$REBUILD" = 0 ] && docker image inspect "$IMAGE:latest" >/dev/null 2>&1; then
  V=$(docker run --rm --entrypoint python3 "$IMAGE:latest" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)
  case "$V" in
    *"${VLLM_REF:0:9}"*) ok "image $IMAGE:latest already contains vllm $V (pinned ${VLLM_REF:0:9}); build skipped. ./install.sh --rebuild to force" ;;
    *) mourir "image $IMAGE:latest exists but its vllm is '$V', not the pinned ${VLLM_REF:0:9}.
       Either retag it out of the way, or ./install.sh --rebuild." ;;
  esac
else
  # 3a. the arm64 build base
  dire "── build 1/3: arm64 build base ($BASE_IMAGE)"
  DOCKER_BUILDKIT=1 docker build -t "$BASE_IMAGE" "$ICI/build/base-arm64" || mourir "base image build failed"
  ok "$BASE_IMAGE"

  # 3b. vLLM from source, at the pinned commit, with vLLM's OWN Dockerfile
  dire "── build 2/3: vLLM at ${VLLM_REF:0:9} (from source; hours)"
  if [ ! -d "$VL/.git" ]; then
    git init -q "$VL" && git -C "$VL" remote add origin "$VLLM_REPO" || mourir "cannot init $VL"
  fi
  A=$(git -C "$VL" rev-parse HEAD 2>/dev/null)
  if [ "$A" != "$VLLM_REF" ]; then
    [ -z "$(git -C "$VL" status --porcelain --untracked-files=no 2>/dev/null)" ] \
      || mourir "upstream-vllm/ is at $A with LOCAL EDITS. Commit or discard them, or delete upstream-vllm/."
    # the commit lives on a PR branch, not on main: fetch it by sha (GitHub allows it)
    git -C "$VL" fetch -q --depth 1 origin "$VLLM_REF" || mourir "cannot fetch vllm $VLLM_REF from $VLLM_REPO"
    git -C "$VL" checkout -q FETCH_HEAD || mourir "cannot check out vllm $VLLM_REF"
  fi
  [ "$(git -C "$VL" rev-parse HEAD)" = "$VLLM_REF" ] || mourir "upstream-vllm/ is not at $VLLM_REF"
  ok "vllm source at $VLLM_REF"
  DOCKER_BUILDKIT=1 docker build -f "$VL/docker/Dockerfile" --target vllm-openai \
    --build-arg BUILD_BASE_IMAGE="$BASE_IMAGE" \
    --build-arg torch_cuda_arch_list=12.0 \
    --build-arg CUDA_VERSION=13.0.3 --build-arg PYTHON_VERSION=3.12 \
    --build-arg GIT_REPO_CHECK=0 \
    -t "$IMAGE-base:latest" "$VL" || mourir "vLLM build failed"
  ok "$IMAGE-base:latest (vLLM ${VLLM_REF:0:9}, target vllm-openai)"

  # 3c. the thin launcher-compatibility layer
  dire "── build 3/3: /workspace + ray $RAY_ATTENDU + empty entrypoint"
  DOCKER_BUILDKIT=1 docker build --build-arg BASE="$IMAGE-base:latest" -t "$IMAGE:latest" "$ICI/build/workspace" \
    || mourir "workspace layer build failed"
  ok "image $IMAGE:latest built"
fi

# ── 4. prove it is the engine the numbers came from.
verifier_moteur_si_possible

derive_des_ancres

dire ""
ok "install complete. Next:"
dire "  1. CHECKPOINT_DIR=... SERVED_DIR=... python3 tools/make-w4a16-variant.py"
dire "  2. copy the checkpoint AND the variant to the worker (both ranks need them)"
dire "  3. ./up.sh --dry-run    then    ./up.sh   under a supervisor, never a shell timeout"
