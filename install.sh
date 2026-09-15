#!/usr/bin/env bash
# One-command landing for capsim on a fresh benchmark host.
#
#   git clone <repo> && cd system-sizing && ./install.sh
#
# Installs uv if missing, then installs capsim as an isolated uv tool
# from this checkout. Idempotent — re-running upgrades in place. No
# system-Python pollution; uninstall with `uv tool uninstall capsim`.
#
# Prerequisites the script checks but does not install: python3.10+,
# docker (needed at run time, not install time), git.
#
# After install:
#   capsim doctor            # validate the host, get a config recommendation
#   capsim smoke --config …  # ~10 min end-to-end pipeline proof (tiny model)
#   capsim ready --config …  # full model download + image pull
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── Prerequisites ────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || die "python3 not found — install Python 3.10+"
python3 - <<'EOF' || die "Python 3.10+ required"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

if ! command -v docker >/dev/null 2>&1; then
  warn "docker not found — capsim installs fine, but engines need Docker at run time (capsim doctor will flag this)"
fi

# ── uv ───────────────────────────────────────────────────────────────
# Remember the PATH the user's shell actually has: we prepend uv's bin
# dir below so THIS script can proceed, but the "is capsim reachable"
# check at the end must answer for the interactive shell, not for us.
PRE_INSTALL_PATH="$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (per-user, no sudo)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv's installer drops into ~/.local/bin (or ~/.cargo/bin on older
  # versions); pick up whichever exists for this shell.
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH — open a new shell and re-run"
fi

# ── capsim ───────────────────────────────────────────────────────────
say "Installing capsim from $REPO_DIR"
uv tool install --force --python 3.12 "$REPO_DIR"

# Check reachability against the PATH the user's shell had BEFORE this
# script prepended uv's bin dir — otherwise the warning never fires
# even though `capsim` will be "command not found" at their prompt.
if ! PATH="$PRE_INSTALL_PATH" command -v capsim >/dev/null 2>&1; then
  uv tool update-shell || true
  warn "capsim is installed but not on your current shell's PATH."
  warn "Run this now (future logins are already fixed by update-shell):"
  warn '    export PATH="$HOME/.local/bin:$PATH"'
fi

say "Installed. Next steps (run from $REPO_DIR so the config/ dir is available):"
echo "    capsim doctor                                  # validate this host"
echo "    capsim smoke --config <config/…yaml>           # ~10 min pipeline proof"
echo "    capsim ready --config <config/…yaml>           # full model + image prep"
