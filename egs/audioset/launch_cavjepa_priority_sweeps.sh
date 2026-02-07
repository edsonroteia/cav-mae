#!/bin/bash
# Launch prioritized CAV-JEPA sweeps.
# Usage:
#   ./launch_cavjepa_priority_sweeps.sh
#   ./launch_cavjepa_priority_sweeps.sh A1 A2b D1

set -euo pipefail
set -x

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

# Default set if no explicit experiment IDs are provided.
experiments=(
  A1
  A2a
  A2b
  D1
  M1
  M2
  C1
  C2
  E2a
  E2b
  E3b
)

if [ "$#" -gt 0 ]; then
  experiments=("$@")
fi

echo "=========================================="
echo "Launching CAV-JEPA Priority Sweeps"
echo "=========================================="
echo "Experiments: ${experiments[*]}"

for exp in "${experiments[@]}"; do
  submit_out=$(sbatch run_cavjepa_ablations.sh "$exp")
  job_id=$(echo "$submit_out" | awk '{print $4}')
  echo "Submitted ${exp}: Job ID ${job_id}"
done

echo "=========================================="
echo "All requested jobs submitted."
echo "Monitor with: squeue -u $USER"
echo "=========================================="
