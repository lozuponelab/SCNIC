#!/bin/bash

set -e

## 
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

## scnic-analysis (main command)
scnic-analysis --help > /dev/null && echo "✓ scnic-analysis CLI works"
scnic-analysis within --help > /dev/null && echo "✓ scnic-analysis within CLI works"
scnic-analysis modules --help > /dev/null && echo "✓ scnic-analysis modules CLI works"
scnic-analysis between --help > /dev/null && echo "✓ scnic-analysis between CLI works"

## module enrichment (idk what this is for yet...)
scnic-module-enrichment --help > /dev/null && echo "✓ scnic-module-enrichment CLI works"
scnic-module-enrichment annotate --help > /dev/null && echo "✓ scnic-module-enrichment annotate CLI works"
scnic-module-enrichment perms --help > /dev/null && echo "✓ scnic-module-enrichment perms CLI works"
scnic-module-enrichment stats --help > /dev/null && echo "✓ scnic-module-enrichment stats CLI works"

## test minimal analysis on test data - only testing within since that is the entry point for this package
scnic-analysis within -i tests/data/fake_data.biom -o /tmp/test_within_out/ -m sparcc > /dev/null && echo "✓ scnic-analysis within: sparcc (default) method CLI works on test data"

echo "smoke tests completed successfully!"