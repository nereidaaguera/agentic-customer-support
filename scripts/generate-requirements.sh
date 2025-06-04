#!/bin/bash
set -e

# Generate requirements.txt
poetry export \
  --format=requirements.txt \
  --output=requirements.txt \
  --without-hashes \
  --only main

# Remove everything from semicolon onwards and deduplicate
python3 << 'EOF'
from pathlib import Path

req_file = Path("requirements.txt")
lines = req_file.read_text().strip().split('\n')

# Clean markers and deduplicate
packages = {}
for line in lines:
    if not line.strip():
        continue
    
    # Remove markers (everything after semicolon)
    clean_line = line.split(' ; ')[0].strip()
    
    # Extract package name
    if '==' in clean_line:
        pkg_name = clean_line.split('==')[0]
        # Keep the last occurrence (Poetry tends to list preferred versions later)
        packages[pkg_name] = clean_line

# Write deduplicated requirements
final_lines = sorted(packages.values())
req_file.write_text('\n'.join(final_lines) + '\n')
print(f"Cleaned and deduplicated to {len(final_lines)} unique packages")
EOF

echo "Generated clean requirements.txt"