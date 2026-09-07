# _config.sh - the ONE place that loads recipe.env. Sourced by install.sh, up.sh, down.sh
#              and view.sh. Do not inline this logic anywhere.
#
# ⚠️ WHY THIS FILE EXISTS AT ALL.
# `recipe.env.local` is an OVERLAY, not a REPLACEMENT: loading only the local file when it
# exists silently drops every variable it does not set (`IMAGE: unbound variable`).
#
# Expects: $ICI = directory of the calling script.
# Provides: every variable of recipe.env, overridden by recipe.env.local where present.

[ -n "${ICI:-}" ] || { echo "FAIL CLOSED: _config.sh sourced without ICI set." >&2; exit 1; }
[ -f "$ICI/recipe.env" ] || { echo "FAIL CLOSED: no recipe.env next to $ICI." >&2; exit 1; }

set -a
. "$ICI/recipe.env"
[ -f "$ICI/recipe.env.local" ] && . "$ICI/recipe.env.local"
set +a

# IMAGE is a repository name; the scripts append ":latest" and "-base:latest" themselves.
case "${IMAGE:-}" in
  *:*|*/*) echo "FAIL CLOSED: IMAGE='$IMAGE' must be a bare name without tag or registry." >&2; exit 1 ;;
esac

# Fail closed on the placeholders, here, once, for all four scripts.
case "${CLUSTER_NODES:-}" in
  ""|*10.0.0.1*) echo "FAIL CLOSED: CLUSTER_NODES is still the placeholder ($CLUSTER_NODES)." >&2
                 echo "  Set the CHANGE ME values in recipe.env, or put them in" >&2
                 echo "  recipe.env.local (an overlay -- it does not replace recipe.env)." >&2
                 exit 1 ;;
esac
