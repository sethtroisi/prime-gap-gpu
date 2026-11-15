#!/usr/bin/env bash
# Builds gpu_sieve_benchmark for the local GPU, runs it, and appends a CSV row.
# Used directly on a machine with a GPU or via colab_benchmark.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CSV_FILE="${CSV_FILE:-$SCRIPT_DIR/sieve_benchmark_results.csv}"
ARCH="${ARCH:-}"
MP="${MP:-}"
TAG="${TAG:-}"
BENCH_BIN="$SCRIPT_DIR/gpu_sieve_benchmark"
DO_BUILD=1
BENCH_ARGS=()
DEFAULT_ARGS=(-p 353 -d 2310 --minc 50000000 --max-prime 1000)

die() { echo "Error: $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [wrapper options] [--] [gpu_sieve_benchmark options]

Builds gpu_sieve_benchmark, runs it (with CPU verification), and appends a CSV row.

Wrapper options:
  --arch <sm_XX>     CUDA arch (default: auto-detect via nvidia-smi)
  --mp <count>       Streaming multiprocessor count (default: auto-detect)
  --tag <label>      Optional label for runs sharing a git hash (or \$TAG)
  -o, --csv <path>   Output CSV path (default: $CSV_FILE)
  --no-build         Skip compilation and run existing binary
  -h, --help         Show this help message

Forwarded gpu_sieve_benchmark options (default: ${DEFAULT_ARGS[*]}):
  -p <p> -d <d> --mstart <m> --minc <m> --max-prime <millions>
  --sieves <n> --ranges <n> --verify <n> --batch <n> --max-pi <n> --first-x <n> --finalize <none|cpu|gpu> --profile
EOF
}

while (($#)); do
  case "$1" in
    --arch=*|--tag=*|--csv=*|--mp=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
    --arch)     [[ $# -ge 2 ]] || die "$1 requires an argument"; ARCH="$2"; shift 2 ;;
    --mp)       [[ $# -ge 2 ]] || die "$1 requires an argument"; MP="$2"; shift 2 ;;
    --tag)      [[ $# -ge 2 ]] || die "$1 requires a label"; TAG="$2"; shift 2 ;;
    -o|--csv)   [[ $# -ge 2 ]] || die "$1 requires a file path"; CSV_FILE="$2"; shift 2 ;;
    --no-build) DO_BUILD=0; shift ;;
    -h|--help)  usage; exit 0 ;;
    --)         shift; BENCH_ARGS+=("$@"); break ;;
    *)          BENCH_ARGS+=("$1"); shift ;;
  esac
done
((${#BENCH_ARGS[@]})) || BENCH_ARGS=("${DEFAULT_ARGS[@]}")

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

detect_arch() {
  local cc
  cc="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" || return 1
  cc="${cc//[[:space:]]/}"
  [[ "$cc" =~ ^([0-9]+)\.([0-9]+)$ ]] || return 1
  echo "sm_${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
}

# GRID_SIZE is a compile time multiple of the number of streaming multiprocessors.
detect_mp() {
  cat > "$TMP_DIR/mp.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 1;
  printf("%d\n", prop.multiProcessorCount);
}
EOF
  nvcc -o "$TMP_DIR/mp" "$TMP_DIR/mp.cu" >/dev/null 2>&1 && "$TMP_DIR/mp"
}

if [[ -z "${GIT_HASH:-}" ]]; then
  if GIT_HASH="$(git -C "$SCRIPT_DIR" rev-parse --short=10 HEAD 2>/dev/null)"; then
    git -C "$SCRIPT_DIR" diff --quiet HEAD -- || GIT_HASH+="-dirty"
  else
    GIT_HASH="unknown"
  fi
fi
echo "Git hash: ${GIT_HASH}${TAG:+ (tag: $TAG)}"

if ((DO_BUILD)); then
  [[ -n "$ARCH" ]] || ARCH="$(detect_arch)" || die "Could not auto-detect GPU arch; specify --arch sm_XX"
  [[ -n "$MP" ]] || MP="$(detect_mp)" || die "Could not auto-detect SM count; specify --mp <count>"
  echo "Compiling gpu_sieve_benchmark (ARCH=$ARCH MP=$MP) ..."
  rm -f "$SCRIPT_DIR"/gpu_sieve.o "$BENCH_BIN"
  make -C "$SCRIPT_DIR" gpu_sieve_benchmark ARCH="$ARCH" MP="$MP" 2>&1 | grep -v '^$' | tail -n 40
else
  ARCH="${ARCH:-prebuilt}"
  [[ -x "$BENCH_BIN" ]] || die "Benchmark binary not found or not executable: $BENCH_BIN"
fi

TMP_OUT="$TMP_DIR/out.txt"
echo "Running $BENCH_BIN ${BENCH_ARGS[*]} ..."
"$BENCH_BIN" "${BENCH_ARGS[@]}" | tee "$TMP_OUT"

CSV_ROW="$(BENCH_TAG="$TAG" awk -v ts="$(date -Iseconds)" -v hash="$GIT_HASH" -v arch="$ARCH" '
  function esc(s) { gsub(/\r/, "", s); gsub(/"/, "\"\"", s); return (s ~ /[,"\n]/) ? "\"" s "\"" : s }
  function val(s, key,   m) { return match(s, key "=[0-9]+") ? substr(s, RSTART + length(key) + 1, RLENGTH - length(key) - 1) : "" }
  BEGIN { tag = esc(ENVIRON["BENCH_TAG"]) }
  /^GPU[ \t]*:/            { sub(/^[^:]*:[ \t]*/, ""); gpu = esc($0) }
  /^GPU SMs[ \t]*:/        { mp = $4 }
  /^Config[ \t]*:/         { p = val($0, "p"); d = val($0, "d"); mstart = val($0, "mstart");
                             minc = val($0, "minc"); max_prime = val($0, "max_prime") }
  /^Sieves[ \t]*:/         { sieves = $3; ranges = $5 }
  /total sieving time/     { gpu_sieves = $7 }
  /^\t[A-Za-z0-9_]+ *: *[0-9.]+ seconds \(/ {
                             name = $1; sub(/:$/, "", name); sec = $0; sub(/^[^:]*: */, "", sec); sub(/ seconds.*/, "", sec);
                             if (gpu_sieves > 0) breakdown = breakdown (breakdown ? ";" : "") name "=" sprintf("%.3f", 1000 * sec / gpu_sieves) }
  /^Batch[ \t]*:/          { batch = $3 }
  /^Finalize[ \t]*:/       { finalize = $3 }
  /^Avg sieve ms[ \t]*:/   { avg_ms = $5; min_ms = $7; median_ms = $10; max_ms = $13 }
  /^Composite[ \t]*:/      { composite = $3 }
  /^Checksum[ \t]*:/       { checksum = $3 }
  /^Verify[ \t]*:/         { verify = $3 }
  END {
    if (avg_ms == "" || checksum == "" || verify == "FAIL" || verify == "") exit 1
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
      ts, hash, tag, gpu, arch, mp, p, d, mstart, minc, max_prime, sieves, ranges,
      batch, avg_ms, min_ms, median_ms, max_ms, esc(breakdown), composite, checksum, verify, finalize
  }
' "$TMP_OUT")" || die "Benchmark failed verification or output could not be parsed"

CSV_HEADER="timestamp,git_hash,tag,gpu,arch,mp,p,d,mstart,minc,max_prime,sieves,ranges,batch,avg_ms,min_ms,median_ms,max_ms,breakdown_ms,composite_pct,checksum,verified,finalize"
mkdir -p "$(dirname "$CSV_FILE")"
if [[ ! -s "$CSV_FILE" ]]; then
  echo "$CSV_HEADER" > "$CSV_FILE"
elif [[ "$(head -n 1 "$CSV_FILE")" != "$CSV_HEADER" ]]; then
  echo "Warning: CSV header in $CSV_FILE does not match expected schema" >&2
fi
printf '%s\n' "$CSV_ROW" >> "$CSV_FILE"
echo "Appended row for git ${GIT_HASH}${TAG:+ (tag: $TAG)} to $CSV_FILE"
