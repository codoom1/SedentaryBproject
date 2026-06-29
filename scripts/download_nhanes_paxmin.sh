#!/bin/bash
set -euo pipefail

OUT_DIR="${1:-/work/pi_jstauden_umass_edu/SBpaper_data_10s/nhanes/paxmin}"
BASE_URL="https://ftp.cdc.gov/pub/NHANES/LargeDataFiles"
CONNECTIONS="${CONNECTIONS:-8}"
CHUNKS_PER_CONNECTION="${CHUNKS_PER_CONNECTION:-2}"

mkdir -p "$OUT_DIR"

download_range() {
  local url="$1"
  local start="$2"
  local end="$3"
  local part="$4"
  local expected=$((end - start + 1))

  if [ -f "$part" ] && [ "$(stat -c %s "$part")" -eq "$expected" ]; then
    echo "[OK] Existing range $start-$end"
    return
  fi

  rm -f "$part"
  echo "[START] Range $start-$end"
  curl --silent --show-error --fail --location \
    --retry 30 --retry-all-errors --retry-delay 5 --connect-timeout 30 \
    --range "$start-$end" --output "$part" "$url"

  local actual
  actual=$(stat -c %s "$part")
  if [ "$actual" -ne "$expected" ]; then
    echo "[ERROR] Range $start-$end has $actual bytes; expected $expected" >&2
    return 1
  fi
  echo "[DONE] Range $start-$end ($actual bytes)"
}

download() {
  local file="$1"
  local expected_bytes="$2"
  local destination="$OUT_DIR/$file"
  local url="$BASE_URL/$file"
  local current_bytes=0

  if [ -f "$destination" ]; then
    current_bytes=$(stat -c %s "$destination")
  fi
  if [ "$current_bytes" -gt "$expected_bytes" ]; then
    echo "[ERROR] $destination is larger than expected; refusing to continue" >&2
    exit 1
  fi
  if [ "$current_bytes" -eq "$expected_bytes" ]; then
    echo "[OK] $file already complete ($current_bytes bytes)"
    return
  fi

  local remaining=$((expected_bytes - current_bytes))
  local chunk_count=$((CONNECTIONS * CHUNKS_PER_CONNECTION))
  local chunk_size=$(((remaining + chunk_count - 1) / chunk_count))
  local part_dir="$OUT_DIR/.${file}.parts.${current_bytes}"
  mkdir -p "$part_dir"

  echo "[INFO] Downloading $file with $CONNECTIONS parallel connections"
  echo "[INFO] Existing=$current_bytes remaining=$remaining expected=$expected_bytes"

  local start end part
  local active=0
  local pids=()
  for ((start=current_bytes; start<expected_bytes; start+=chunk_size)); do
    end=$((start + chunk_size - 1))
    if [ "$end" -ge "$expected_bytes" ]; then
      end=$((expected_bytes - 1))
    fi
    part=$(printf "%s/%020d.part" "$part_dir" "$start")
    download_range "$url" "$start" "$end" "$part" &
    pids+=("$!")
    active=$((active + 1))
    if [ "$active" -ge "$CONNECTIONS" ]; then
      for pid in "${pids[@]}"; do
        wait "$pid"
      done
      active=0
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  echo "[INFO] Combining downloaded ranges for $file"
  for part in "$part_dir"/*.part; do
    cat "$part" >> "$destination"
  done

  local actual_bytes
  actual_bytes=$(stat -c %s "$destination")
  if [ "$actual_bytes" -ne "$expected_bytes" ]; then
    echo "[ERROR] $file has $actual_bytes bytes; expected $expected_bytes" >&2
    exit 1
  fi
  rm -rf "$part_dir"
  echo "[OK] $file ($actual_bytes bytes)"
}

download "PAXMIN_G.xpt" 8125196000  # NHANES 2011-2012
download "PAXMIN_H.xpt" 9351691760  # NHANES 2013-2014

echo "[DONE] NHANES minute-level PAM files saved under $OUT_DIR"
