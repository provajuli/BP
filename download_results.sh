#!/usr/bin/env bash
set -euo pipefail

LOGIN="xprova06"
REMOTE_HOST="eva.fit.vutbr.cz"
REMOTE_DIR="/homes/eva/xp/${LOGIN}/WWW/results"

BASE="$HOME/a_school/BP/bp_results"
RAW="$BASE/raw"
PROCESSED="$BASE/processed"
EXCLUDED="$BASE/excluded"

mkdir -p "$RAW" "$PROCESSED" "$EXCLUDED"

# socket pro sdílené SSH spojení
CONTROL_PATH="/tmp/ssh_mux_${LOGIN}@${REMOTE_HOST}"

cleanup() {
  ssh -O exit \
    -o ControlPath="$CONTROL_PATH" \
    "${LOGIN}@${REMOTE_HOST}" 2>/dev/null || true
}
trap cleanup EXIT

# otevři jedno master spojení -> heslo zadáš jen jednou
ssh -fnN \
  -o ControlMaster=yes \
  -o ControlPath="$CONTROL_PATH" \
  -o ControlPersist=10m \
  "${LOGIN}@${REMOTE_HOST}"

# seznam CSV na serveru
ssh \
  -o ControlMaster=auto \
  -o ControlPath="$CONTROL_PATH" \
  "${LOGIN}@${REMOTE_HOST}" \
  "ls ${REMOTE_DIR}/*.csv 2>/dev/null" | while read -r remote_file; do

  name="$(basename "$remote_file")"

  if [[ -e "$RAW/$name" || -e "$PROCESSED/$name" || -e "$EXCLUDED/$name" ]]; then
    echo "Skipping $name (already exists)"
    continue
  fi

  echo "Downloading $name"
  rsync -av \
    -e "ssh -o ControlMaster=auto -o ControlPath=$CONTROL_PATH" \
    "${LOGIN}@${REMOTE_HOST}:${remote_file}" \
    "$RAW/"
done