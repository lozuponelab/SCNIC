#!/bin/bash

set -e

## print scnic version so i can see if its being pulled from the git tags correctly
scnicVersion=$( scnic --version 2>&1 )
echo "current scnic version: ${scnicVersion}"

echo "running smoke tests for scnic..."

python -c "import SCNIC; print('✓ scnic imported')"

## test cli help commands
## scnic (main command) - same as scnic-analysis, kept in next version for consistency for how the main
## analysis was called in the past 
scnic --help > /dev/null && echo "✓ scnic --help CLI works"
scnic --version > /dev/null && echo "✓ scnic --version CLI works"
scnic within --help > /dev/null && echo "✓ scnic within CLI works"
scnic modules --help > /dev/null && echo "✓ scnic modules CLI works"
scnic between --help > /dev/null && echo "✓ scnic between CLI works"

## module enrichment (idk what this is for yet...)
scnic-module-enrichment --help > /dev/null && echo "✓ scnic-module-enrichment CLI works"
scnic-module-enrichment annotate --help > /dev/null && echo "✓ scnic-module-enrichment annotate CLI works"
scnic-module-enrichment perms --help > /dev/null && echo "✓ scnic-module-enrichment perms CLI works"
scnic-module-enrichment stats --help > /dev/null && echo "✓ scnic-module-enrichment stats CLI works"

## test minimal analysis on test data - only testing within since that is the entry point for this package
scnic within -i tests/data/fake_data.biom -o /tmp/test_within_out/ -m sparcc > /dev/null && echo "✓ scnic within: sparcc (default) method CLI works on test data"
## idk if this will work but fingers crossed 
scnic modules -i /tmp/test_within_out/within_sparcc_correls.txt -o /tmp/test_modules_out/ --method naive --min_r 0.35  --table tests/data/fake_data.biom > /dev/null && echo "✓ scnic modules: naive (default) method CLI works on test data"

echo "smoke tests completed successfully!"