#!/bin/bash

# find all parquet files in the current directory and subdirectories
find . -type f -name "*_events.parquet" | while read -r file; do
		# extract the directory name and file name without extension
		dirname=$(dirname "$file")
		filename=$(basename "$file" .parquet)
		output_file="${dirname}/${filename}.h5"
		filename_base=$(basename "$file" _events.parquet)
		particles_file="${dirname}/${filename_base}_particles.parquet"
		# call the Python script to convert the file
		if [ ! -f "$particles_file" ]; then
			echo "Particles file not found for $file, skipping conversion."
			continue
		fi
		echo "Converting $file to $output_file"
		./parquet2h5.py --parquet-events "$file" --parquet-particles "$particles_file" --h5output-filename "$output_file"
		if [ $? -eq 0 ]; then
			echo "Converted $file to $output_file"
		else
			echo "Failed to convert $file"
			break
		fi
done