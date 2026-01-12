#!/usr/bin/env bash
set -euo pipefail

LOGIN="xprova06"
REMOTE_HOST="eva.fit.vutbr.cz"
REMOTE_DIR="/homes/eva/xp/${LOGIN}/WWW/results/"
LOCAL_DIR="$HOME/a_school/BP/bp_results/raw"

mkdir -p "$LOCAL_DIR"

# stáhne jen nové soubory (už existující nepřepisuje)
rsync -av --ignore-existing "${LOGIN}@${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR/"

echo "Downloaded into: $LOCAL_DIR"
