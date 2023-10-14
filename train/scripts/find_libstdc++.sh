#!/bin/bash
files=$(find / -name 'libstdc++.so.6*' 2>/dev/null)
total=$(echo "$files" | wc -l)
counter=1

echo "$total files found."

for file in $files; do
    echo "Processing file $counter/$total: $file"
    if strings "$file" 2>/dev/null | grep -q 'CXXABI_1.3.9'; then
        echo "File matches: $file"
    fi
    ((counter++))
done
