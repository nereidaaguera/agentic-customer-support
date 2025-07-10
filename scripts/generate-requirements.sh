#!/bin/bash
set -e

# Generate requirements.txt using poetry show
echo "Generating requirements.txt from Poetry dependencies..."

# Get all main dependencies with their versions
poetry show --only main --no-ansi | awk '{print $1"=="$2}' > requirements.txt

# Clean markers, deduplicate, and filter platform-specific packages
python3 << 'EOF'
from pathlib import Path

req_file = Path("requirements.txt")
lines = req_file.read_text().strip().split('\n')

# Packages to exclude for Databricks compatibility
EXCLUDED_PACKAGES = {
    # Windows-specific packages
    'pywin32',
    'colorama',
    'waitress',
    
    # Pre-installed in Databricks Runtime - causes conflicts if reinstalled
    'databricks-connect',
    'databricks-cli',
    'py4j',
}

packages = {}
for line in lines:
    if not line.strip():
        continue
    
    # Check if this line has platform markers
    if ' ; ' in line:
        clean_line, markers = line.split(' ; ', 1)
        
        # Skip Windows-specific packages
        if 'platform_system == "Windows"' in markers or 'sys_platform == "win32"' in markers:
            continue
    else:
        clean_line = line.strip()
    
    if '==' in clean_line:
        pkg_name = clean_line.split('==')[0]
        
        if pkg_name in EXCLUDED_PACKAGES:
            continue
            
        # Keep the last occurrence
        packages[pkg_name] = clean_line

# Write deduplicated requirements
final_lines = sorted(packages.values())
req_file.write_text('\n'.join(final_lines) + '\n')
print(f"Cleaned and deduplicated to {len(final_lines)} unique packages")
EOF

echo "Generated requirements.txt"