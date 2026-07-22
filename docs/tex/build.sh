#!/usr/bin/env bash
set -eu

target="$1"

latexmk \
  -jobname="$target" \
  "contents/$target/main.tex"
