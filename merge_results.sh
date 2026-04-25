#!/usr/bin/env bash
set -euo pipefail

PROC_DIR="$HOME/a_school/BP/bp_results/processed"
OUT="$HOME/a_school/BP/bp_results/master.csv"

mkdir -p "$(dirname "$OUT")"

files=( "$PROC_DIR"/*.csv )
if [ ! -e "${files[0]}" ]; then
  echo "No CSV files in $PROC_DIR"
  exit 0
fi

# vyčisti output
: > "$OUT"

# hlavička z prvního souboru
head -n 1 "${files[0]}" > "$OUT"

# data ze všech souborů
for f in "${files[@]}"; do
  echo "# $(basename "$f")" >> "$OUT"
  tail -n +2 "$f" >> "$OUT"
done

echo "Merged into: $OUT"
echo "Rows (incl. header): $(wc -l < "$OUT")"
