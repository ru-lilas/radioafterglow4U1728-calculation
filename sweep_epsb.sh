#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 INDEX [--combine]" >&2
  exit 1
fi

index="$1"
echo "index: $index"

eps_b_values=$(python -c 'import numpy as np; print(*np.geomspace(1e-3, 1e-1, 8))')

python -m scripts.build_input \
  --mode eps-b-sweep \
  --template input/general.yaml \
  --runs-dir data \
  --eps-b-values $eps_b_values 0.02\
  --sweep-index "data/${index}.csv" \
  --yes

if [[ "${2:-}" == "--combine" ]]; then
  python -m scripts.combine_energetics \
    --sweep-index "data/${index}.csv" \
    --output "data/budget_${index}.csv"

  make "BUDGET=budget_${index}" "fig/budget_${index}.pdf"
fi

