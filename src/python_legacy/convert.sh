#!/usr/bin/env bash
set -euo pipefail

# === Customizable Section ===
NB_DIR="notebooks"      # Folder containing .ipynb files (relative to this .sh)
OUT_DIR="modules"       # Output folder for .py files
# Manually specify the .ipynb files to convert here (filenames can include spaces):
#   List one per line, use a backslash at the end for line continuation
NOTEBOOKS=(
    "constants.ipynb"
    "utils.ipynb"
    "calculations.ipynb"
    "LinearShallowEquatorialWave.ipynb"
    "plotter.ipynb"
    # "test.ipynb"
)

# === Usually no need to change below ===
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
NB_ABS="${ROOT_DIR}/${NB_DIR}"
OUT_ABS="${ROOT_DIR}/${OUT_DIR}"

# Convert notebooks
for nb in "${NOTEBOOKS[@]}"; do
  SRC="${NB_ABS}/${nb}"
  if [[ ! -f "$SRC" ]]; then
    echo "⚠️ File not found: $SRC (skipped)"
    continue
  fi
  echo "➡️ Converting: $SRC"
  jupyter nbconvert \
    --to script \
    --log-level=ERROR \
    --TemplateExporter.exclude_input_prompt=True \
    --TemplateExporter.exclude_output_prompt=True \
    "$SRC" \
    --output-dir "$OUT_ABS"
  # Alternative:
  # jupyter nbconvert --to script "$SRC" --output-dir "$OUT_ABS"
done

echo "✅ All done, output saved in: $OUT_ABS"
