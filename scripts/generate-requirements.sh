#!/bin/bash
set -e

# Generate requirements.txt
poetry export \
  --format=requirements.txt \
  --output=requirements.txt \
  --without-hashes \
  --only main

# Remove everything from semicolon onwards (macOS compatible)
sed -i '' 's/ ; .*//' requirements.txt

echo "Generated clean requirements.txt"