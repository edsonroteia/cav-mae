#!/bin/bash
# =============================================================================
# Aggregate all retrieval results into summary tables
# =============================================================================

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

RESULTS_BASE=./exp/retrieval_results
OUTPUT_FILE="${RESULTS_BASE}/all_results_summary.csv"

echo "=============================================="
echo "Aggregating Retrieval Results"
echo "=============================================="

# Create header
echo "model,sweep,direction,R@1,R@5,R@10,MR" > ${OUTPUT_FILE}

# Function to process a directory
process_dir() {
    local dir=$1
    local sweep_name=$2

    if [ -d "$dir" ]; then
        for csv in ${dir}/*.csv; do
            if [ -f "$csv" ] && [ "$(basename $csv)" != "summary.csv" ]; then
                model_name=$(basename $csv .csv)
                # Skip header and add sweep name
                tail -n +2 $csv 2>/dev/null | while read line; do
                    echo "${model_name},${sweep_name},${line}"
                done
            fi
        done
    fi
}

# Process base results
process_dir "${RESULTS_BASE}" "base" >> ${OUTPUT_FILE}

# Process sweep directories
process_dir "${RESULTS_BASE}/orthogonal-sweep" "orthogonal" >> ${OUTPUT_FILE}
process_dir "${RESULTS_BASE}/weighted-sweep" "weighted" >> ${OUTPUT_FILE}
process_dir "${RESULTS_BASE}/task-arithmetic-sweep" "task-arithmetic" >> ${OUTPUT_FILE}
process_dir "${RESULTS_BASE}/dare-ties-sweep" "dare-ties" >> ${OUTPUT_FILE}

echo ""
echo "Results saved to: ${OUTPUT_FILE}"
echo ""

# Count results
total=$(tail -n +2 ${OUTPUT_FILE} | wc -l)
echo "Total result rows: $total"
echo ""

# Show top results by R@1 (audio to visual)
echo "=============================================="
echo "Top 10 Models by R@1 (Audio -> Visual)"
echo "=============================================="
echo "model,sweep,direction,R@1,R@5,R@10,MR"
grep ",a2v," ${OUTPUT_FILE} | sort -t',' -k4 -rn | head -10

echo ""
echo "=============================================="
echo "Top 10 Models by R@1 (Visual -> Audio)"
echo "=============================================="
echo "model,sweep,direction,R@1,R@5,R@10,MR"
grep ",v2a," ${OUTPUT_FILE} | sort -t',' -k4 -rn | head -10

echo ""
echo "Full results in: ${OUTPUT_FILE}"
