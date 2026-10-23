#!/usr/bin/env bash
# Usage: curl -fsSL https://raw.githubusercontent.com/cpaxton/trace-transport/main/scripts/install.sh | bash
set -euo pipefail

usage() {
    echo 'Usage: install.sh [--dir PATH] [--baseline-only]'
    echo 'Default: clone into ./trace-transport and install learned-policy dependencies.'
}
trace_dir="${TRACE_TRANSPORT_DIR:-$PWD/trace-transport}"
trace_learn=1
while (($#)); do
    case "$1" in
        --dir) [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }; trace_dir="$2"; shift 2 ;;
        --baseline-only) trace_learn=0; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
for trace_command in git curl; do
    command -v "$trace_command" >/dev/null || { echo "Install $trace_command first, then retry." >&2; exit 1; }
done
if ! command -v uv >/dev/null; then
    trace_bootstrap=$(mktemp)
    trap 'rm -f "$trace_bootstrap"' EXIT
    curl --fail --show-error --silent --location https://astral.sh/uv/install.sh -o "$trace_bootstrap"
    sh "$trace_bootstrap"
    export PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null || { echo 'uv installation finished but uv is not on PATH.' >&2; exit 1; }
if [[ -e "$trace_dir" ]]; then
    [[ -d "$trace_dir/.git" || -f "$trace_dir/.git" ]] || { echo "Destination exists and is not a checkout: $trace_dir" >&2; exit 1; }
    trace_origin=$(git -C "$trace_dir" remote get-url origin)
    case "$trace_origin" in
        https://github.com/cpaxton/trace-transport.git|https://github.com/cpaxton/trace-transport|git@github.com:cpaxton/trace-transport.git) ;;
        *) echo "Destination belongs to another repository: $trace_dir" >&2; exit 1 ;;
    esac
    echo "Using existing checkout without updating or resetting it: $trace_dir"
else
    git clone https://github.com/cpaxton/trace-transport.git "$trace_dir"
fi
cd "$trace_dir"
trace_uv_args=(--locked)
if ((trace_learn)); then trace_uv_args+=(--extra learn); fi
uv sync "${trace_uv_args[@]}"
uv run "${trace_uv_args[@]}" trace-transport --help >/dev/null
trace_bin_dir="${TRACE_TRANSPORT_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$trace_bin_dir"
trace_launcher="$trace_bin_dir/trace-transport"
trace_executable="$PWD/.venv/bin/trace-transport"
if [[ -e "$trace_launcher" || -L "$trace_launcher" ]]; then
    if [[ ! -L "$trace_launcher" || "$(readlink "$trace_launcher")" != "$trace_executable" ]]; then
        echo "Refusing to overwrite existing command: $trace_launcher" >&2
        echo 'Choose another TRACE_TRANSPORT_BIN_DIR or remove the old installation explicitly.' >&2
        exit 1
    fi
else
    ln -s "$trace_executable" "$trace_launcher"
fi
printf '\nInstalled command: %s\n' "$trace_launcher"
case ":$PATH:" in
    *":$trace_bin_dir:"*) ;;
    *) printf 'Add this to your shell profile, then reopen the terminal:\n  export PATH=%q:"$PATH"\n' "$trace_bin_dir" ;;
esac
printf '\nStart the test bench from the checkout:\n  cd %q\n  trace-transport console --open\n' "$PWD"
