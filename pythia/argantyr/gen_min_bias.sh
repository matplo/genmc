#!/bin/bash

# Script to generate multiple PYTHIA samples with different random seeds in parallel
# Usage: ./gen_min_bias.sh <cmnd_file> <m_samples> <n_events_per_sample> [output_directory]

# Check if gnu parallel is available
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is not installed. Please install it first."
    echo "On Ubuntu/Debian: sudo apt install parallel"
    echo "On RHEL/CentOS: sudo yum install parallel"
    exit 1
fi

# Check if pythia_parquet.py exists
if [ ! -f "pythia_parquet.py" ]; then
    echo "Error: pythia_parquet.py not found in current directory"
    exit 1
fi

# Parse command line arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <cmnd_file> <m_samples> <n_events_per_sample> [output_directory]"
    echo "  cmnd_file: Path to PYTHIA command file"
    echo "  m_samples: Number of samples to generate"
    echo "  n_events_per_sample: Number of events per sample"
    echo "  output_directory: Optional output directory (default: basename of cmnd_file)"
    exit 1
fi

CMND_FILE="$1"
M_SAMPLES="$2"
N_EVENTS="$3"
OUTPUT_DIR="$4"

# Check if command file exists
if [ ! -f "$CMND_FILE" ]; then
    echo "Error: Command file '$CMND_FILE' not found"
    exit 1
fi

# Validate numeric arguments
if ! [[ "$M_SAMPLES" =~ ^[0-9]+$ ]] || [ "$M_SAMPLES" -lt 1 ]; then
    echo "Error: m_samples must be a positive integer"
    exit 1
fi

if ! [[ "$N_EVENTS" =~ ^[0-9]+$ ]] || [ "$N_EVENTS" -lt 1 ]; then
    echo "Error: n_events_per_sample must be a positive integer"
    exit 1
fi

# Set output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=$(basename "$CMND_FILE" .cmnd)
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Calculate number of cores to use (ncores/2)
NCORES=$(nproc)
MAX_JOBS=$((NCORES / 2))
if [ "$MAX_JOBS" -lt 1 ]; then
    MAX_JOBS=1
fi

echo "Configuration:"
echo "  Command file: $CMND_FILE"
echo "  Number of samples: $M_SAMPLES"
echo "  Events per sample: $N_EVENTS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Available cores: $NCORES"
echo "  Parallel jobs: $MAX_JOBS"
echo

# Function to generate a single sample
generate_sample() {
    local seed=$1
    local cmnd_file=$2
    local n_events=$3
    local output_dir=$4
    
    # Create output filename with seed
    local base_name=$(basename "$cmnd_file" .cmnd)
    local output_file_base="${output_dir}/seed_${seed}/${base_name}_seed_${seed}.parquet"
	
		mkdir -p "$(dirname "$output_file_base")"
    # Check if files already exist
    if [ -f "$output_file_base" ]; then
        echo "Seed $seed: Files already exist, skipping..."
        return 0
    fi
    
    echo "Seed $seed: Generating $n_events events..."
    
    # Run pythia_parquet.py with specific seed and output files
    python3 pythia_parquet.py \
        --cmnd "$cmnd_file" \
        --nev "$n_events" \
        --seed "$seed" \
        --output "$output_file_base"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Seed $seed: Successfully generated files"
    else
        echo "Seed $seed: Failed with exit code $exit_code"
        # Clean up incomplete files
        rm -f "$events_file" "$particles_file"
    fi
    
    return $exit_code
}

# Export function and variables for parallel
export -f generate_sample
export CMND_FILE N_EVENTS OUTPUT_DIR

# Generate sequence of random seeds
echo "Generating $M_SAMPLES samples with random seeds..."

# Create a list of random seeds
SEEDS=$(python3 -c "
import random
random.seed(42)  # For reproducibility
seeds = [random.randint(1, 1000000) for _ in range($M_SAMPLES)]
print(' '.join(map(str, seeds)))
")

# Run parallel generation
echo "$SEEDS" | tr ' ' '\n' | parallel -j "$MAX_JOBS" generate_sample {} "$CMND_FILE" "$N_EVENTS" "$OUTPUT_DIR"

echo
echo "Generation complete!"
echo "Output files are in: $OUTPUT_DIR"
echo "Files generated:"
find "$OUTPUT_DIR" -name "*.parquet" 2>/dev/null || echo "No parquet files found"
