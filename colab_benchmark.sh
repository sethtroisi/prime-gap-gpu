#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOCAL_CSV="${CSV_FILE:-$REPO_ROOT/sieve_benchmark_results.csv}"
SESSION="${SESSION:-}"
GPU="${GPU:-L4}"
ARCH="${ARCH:-}"
MP="${MP:-}"
TAG="${TAG:-}"
STOP_AFTER=0
BENCH_ARGS=()

die() { echo "Error: $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [--] [gpu_sieve_benchmark options]

Uploads the repository (tracked + untracked, non-ignored files) to a Colab GPU
session, compiles and runs gpu_sieve_benchmark (see run_sieve_benchmark.sh),
and appends verified benchmark results to a local CSV file.

Options:
  -s, --session <name>   Colab session to reuse or create (default: active GPU or 'sieve-bench')
  --gpu <type>           GPU when creating a session: T4, L4, G4, A100, H100 (default: $GPU)
  --arch <sm_XX>         CUDA arch override (default: auto-detect)
  --mp <count>           Streaming multiprocessor count override (default: auto-detect)
  --tag <label>          Optional label for runs sharing a git hash (or \$TAG)
  -o, --csv <path>       Local CSV file to append results to (default: $LOCAL_CSV)
  --stop                 Stop the Colab session on exit if created by this run
  -h, --help             Show this help message

Forwarded gpu_sieve_benchmark options (default: -p 353 -d 2310 --minc 50000000 --max-prime 1000):
  -p <p> -d <d> --mstart <m> --minc <m> --max-prime <millions>
  --sieves <n> --ranges <n> --verify <n> --batch <n> --max-pi <n> --first-x <n> --finalize <none|cpu|gpu> --profile
EOF
}

while (($#)); do
  case "$1" in
    --session=*|--gpu=*|--arch=*|--mp=*|--tag=*|--csv=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
    -s|--session) [[ $# -ge 2 ]] || die "$1 requires a session name"; SESSION="$2"; shift 2 ;;
    --gpu)        [[ $# -ge 2 ]] || die "$1 requires a GPU type"; GPU="$2"; shift 2 ;;
    --arch)       [[ $# -ge 2 ]] || die "$1 requires sm_XX"; ARCH="$2"; shift 2 ;;
    --mp)         [[ $# -ge 2 ]] || die "$1 requires a count"; MP="$2"; shift 2 ;;
    --tag)        [[ $# -ge 2 ]] || die "$1 requires a label"; TAG="$2"; shift 2 ;;
    -o|--csv)     [[ $# -ge 2 ]] || die "$1 requires a CSV path"; LOCAL_CSV="$2"; shift 2 ;;
    --stop)       STOP_AFTER=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; BENCH_ARGS+=("$@"); break ;;
    *)            BENCH_ARGS+=("$1"); shift ;;
  esac
done

if [[ -n "${COLAB_BIN:-}" ]]; then COLAB=("$COLAB_BIN")
elif command -v colab >/dev/null 2>&1; then COLAB=(colab)
elif [[ -x "$HOME/.venv/bin/colab" ]]; then COLAB=("$HOME/.venv/bin/colab")
elif command -v uv >/dev/null 2>&1 && uv run colab version >/dev/null 2>&1; then COLAB=(uv run colab)
else die "'colab' CLI not found (install with: uv tool install google-colab-cli)"
fi

TMP_DIR="$(mktemp -d)"
CREATED_SESSION=0
cleanup() {
  rm -rf "$TMP_DIR"
  if ((STOP_AFTER && CREATED_SESSION)) && [[ -n "$SESSION" ]]; then
    echo "Stopping Colab session '$SESSION' ..."
    "${COLAB[@]}" stop -s "$SESSION" || true
  fi
}
trap cleanup EXIT

# 1. Compute local git hash (carried over to Colab since .git is not tarred)
GIT_HASH="$(git -C "$REPO_ROOT" rev-parse --short=10 HEAD)"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain -- . ':!sieve_benchmark_results.csv' ':!sieve_notes.md')" ]] || GIT_HASH+="-dirty"

# 2. Tar all git-tracked and untracked (but not ignored) files in the repository
LOCAL_TAR="$TMP_DIR/prime-gap-gpu.tar.gz"
git -C "$REPO_ROOT" ls-files -z --cached --others --exclude-standard | while IFS= read -r -d '' f; do
  [[ ! -e "$REPO_ROOT/$f" ]] || printf '%s\0' "$f"
done | tar -C "$REPO_ROOT" --null -T - -czf "$LOCAL_TAR"

# 3. Find an active Colab GPU session or create a new one
SESSIONS_OUT="$("${COLAB[@]}" sessions)"
if [[ -z "$SESSION" ]]; then
  SESSION="$(awk '/^\[[^?][^]]*\].*\| Variant: GPU(\r|$| )/ { sub(/^\[/, ""); sub(/\].*/, ""); print; exit }' <<<"$SESSIONS_OUT")"
fi
if [[ -n "$SESSION" ]] && grep -Fq "[$SESSION] " <<<"$SESSIONS_OUT"; then
  echo "Using Colab session '$SESSION'"
else
  SESSION="${SESSION:-sieve-bench}"
  echo "Creating Colab session '$SESSION' (--gpu $GPU) ..."
  "${COLAB[@]}" new -s "$SESSION" --gpu "$GPU"
  CREATED_SESSION=1
fi

# 4. Upload archive to Colab
REMOTE_TAR="content/prime-gap-gpu.tar.gz"
REMOTE_CSV="content/sieve_benchmark.csv"
echo "Uploading archive ($(wc -c < "$LOCAL_TAR" | tr -d '[:space:]') bytes) and running on '$SESSION' ..."
"${COLAB[@]}" upload -s "$SESSION" "$LOCAL_TAR" "$REMOTE_TAR"

# 5. Untar, compile, and run the benchmark binary on Colab
"${COLAB[@]}" ssh -s "$SESSION" -- \
  bash -s -- "$GIT_HASH" "$TAG" "$ARCH" "$MP" "/$REMOTE_CSV" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} <<'EOF'
set -euo pipefail
export GIT_HASH="$1" TAG="$2" ARCH="$3" MP="$4" PATH="/usr/local/cuda/bin:$PATH"
REMOTE_CSV="$5"; shift 5
rm -rf /content/prime-gap-gpu "$REMOTE_CSV"
mkdir -p /content/prime-gap-gpu && tar -xzf /content/prime-gap-gpu.tar.gz -C /content/prime-gap-gpu
if ! printf '#include <gmp.h>\n#include <primesieve.hpp>\n' | g++ -E -x c++ - >/dev/null 2>&1; then
  echo "Installing libgmp-dev libprimesieve-dev on Colab VM ..."
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgmp-dev libprimesieve-dev >/dev/null
fi
bash /content/prime-gap-gpu/run_sieve_benchmark.sh --csv "$REMOTE_CSV" -- "$@"
EOF

DOWNLOADED_CSV="$TMP_DIR/results.csv"
"${COLAB[@]}" download -s "$SESSION" "$REMOTE_CSV" "$DOWNLOADED_CSV"
[[ -s "$DOWNLOADED_CSV" ]] || die "Downloaded CSV is empty"

mkdir -p "$(dirname "$LOCAL_CSV")"
if [[ ! -s "$LOCAL_CSV" ]]; then
  cp "$DOWNLOADED_CSV" "$LOCAL_CSV"
else
  [[ "$(head -n 1 "$LOCAL_CSV")" == "$(head -n 1 "$DOWNLOADED_CSV")" ]] ||
    echo "Warning: CSV header in $LOCAL_CSV does not match downloaded header" >&2
  tail -n +2 "$DOWNLOADED_CSV" >> "$LOCAL_CSV"
fi
echo "Appended $(($(wc -l < "$DOWNLOADED_CSV") - 1)) row(s) from '$SESSION'${TAG:+ (tag: $TAG)} to $LOCAL_CSV"
