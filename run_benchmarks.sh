#!/bin/bash

# Usage: ./run_benchmarks.sh fs_0045_3T.h5

INPUT_FILE=${1:-"us_0001_3t.h5"}
OUTPUT_BASE="benchmarks/$(basename "$INPUT_FILE" .h5)"

echo "Starting RD Curve generation for $INPUT_FILE..."

python run_rd_curve.py --input "$INPUT_FILE" --output_dir "$OUTPUT_BASE"

echo "Benchmarking complete. Plots available in $OUTPUT_BASE"