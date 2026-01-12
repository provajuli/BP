#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="$HOME/a_school/BP/bp_results/raw"
OUT="$HOME/a_school/BP/bp_results/master.csv"

mkdir -p "$(dirname "$OUT")"

# najdi csv, seřaď, vezmi první hlavičku, pak appenduj zbytek bez hlaviček
files=( "$RAW_DIR"/*.csv )
if [ ! -e "${files[0]}" ]; then
  echo "No CSV files in $RAW_DIR"
  exit 0
fi

# vyčisti output
: > "$OUT"

# hlavička z prvního souboru
head -n 1 "${files[0]}" > "$OUT"

# data ze všech souborů bez hlaviček
for f in "${files[@]}"; do
  tail -n +2 "$f" >> "$OUT"
done

echo "Merged into: $OUT"
echo "Rows (incl. header): $(wc -l < "$OUT")"
