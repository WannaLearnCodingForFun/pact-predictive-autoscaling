#!/bin/sh
# docx -> pdf without image downsampling or lossy recompression.
set -e
IN="$1"; OUT="${2:-/tmp/pdfout}"
mkdir -p "$OUT"
soffice --headless --convert-to \
 'pdf:writer_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"},"MaxImageResolution":{"type":"long","value":"1200"},"ExportBookmarks":{"type":"boolean","value":"true"}}' \
 --outdir "$OUT" "$IN" >/dev/null 2>&1
echo "$OUT/$(basename "${IN%.docx}").pdf"
